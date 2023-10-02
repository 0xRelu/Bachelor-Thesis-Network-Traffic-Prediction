import abc
import concurrent.futures
import csv
import itertools
import multiprocessing
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from scapy.layers.inet import TCP, UDP, IP, ICMP
from scapy.packet import Packet
from scapy.plist import PacketList
from scapy.utils import rdpcap


def _split_flows_(packets: PacketList, max_flows: int) -> list:
    data_flows = {}

    counter = 0

    for packet in packets:
        counter += 1

        if IP not in packet or (UDP not in packet[IP] and TCP not in packet[IP]):
            continue

        flow_id_1 = packet[IP].src + "|" + packet[IP].dst
        flow_id_2 = packet[IP].dst + "|" + packet[IP].src

        flow_id = flow_id_1

        if flow_id_1 not in data_flows \
                and flow_id_2 not in data_flows \
                and len(data_flows) >= max_flows > 0:
            continue

        # add new flow
        if flow_id_1 not in data_flows and flow_id_2 not in data_flows:
            data_flows[flow_id_1] = []
            if len(data_flows) % 10 == 0:
                print(f"[+] \t Added new flow: {len(data_flows)}")

        if flow_id_2 in data_flows:
            flow_id = flow_id_2

        packet_time = float(packet.time)
        size = len(packet)

        # add new entry to flow
        data_flows.get(flow_id).append(
            [packet_time, size, 0 if flow_id_1 is flow_id else 1, hot_embed_protocol(packet)])

        if counter % 5000 == 0:
            print(f"[+] Packets loaded: {counter / len(packets)}")

    return [np.array(flow) for flow in data_flows.values()]


def hot_embed_protocol(packet: Packet) -> int:
    if IP not in packet:
        raise IndexError("IP has to be in packet.")

    if TCP in packet[IP]:
        return 0

    if UDP in packet[IP]:
        return 1

    raise IndexError(f"Protocol not defined {packet.summary()}")


def split_list(input_list, num_parts):
    avg = len(input_list) / float(num_parts)
    out = []
    last = 0.0

    while last < len(input_list):
        out.append(input_list[int(last):int(last + avg)])
        last += avg

    return out


def parse_pcap_to_list(file_path: str, save_path: str):
    print(f"[+] Loading packets from pcap file with location: {file_path} ...")
    packetList = rdpcap(file_path)
    packetList = _split_flows_(packetList, -1)

    with open(save_path, 'wb') as f:
        pickle.dump(packetList, f)

    print(f"[+] Wrote {len(packetList)} flows successfully in file with path {save_path}")


def _list_milliseconds_(data_flow: np.ndarray, aggregation_time: int = 1000) -> list:
    start_time = int(data_flow[0][0] * aggregation_time)
    end_time = int(data_flow[-1][0] * aggregation_time) + 1
    flow = [[time / aggregation_time, []] for time in range(start_time, end_time + 1)]

    for packet in data_flow:
        packet_time = int(packet[0] * aggregation_time)
        flow[packet_time - start_time][1].append(packet)

    return flow


def _list_milliseconds_only_sizes_(data_flow: ndarray, aggregation_time: int = 1000) -> np.ndarray:
    start_time = int(data_flow[0][0] * aggregation_time)  # assumes packets are ordered
    end_time = int(data_flow[-1][0] * aggregation_time) + 1
    flow = [[time / aggregation_time, 0] for time in range(start_time, end_time + 1)]

    for packet in data_flow:
        packet_time = int(packet[0] * aggregation_time)
        flow[packet_time - start_time][1] += packet[1]

    return np.array(flow)


class DataTransformerBase:
    def __init__(self, file_path: str):
        self.file_path = file_path

        if not file_path.endswith('.pkl'):
            raise AttributeError('Muss ein phl file sein. Wenn es noch ein pcap file ist. Parse mit Methode!')

        self.packets = self._load_packet_from_pkl(file_path)
        self.data = self._get_data(data_flows=self.packets)

    def _load_packet_from_pkl(self, file_path) -> list:
        print(f"[+] Loading packets from pkl file with location: {self.file_path} ...")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        return data

    def save_python_object(self, py_save_path: str) -> str:
        with open(py_save_path, 'wb') as f:
            pickle.dump(self.data, f)

        print(f"[+] Wrote {len(self.data)} successfully in file with path {py_save_path}")
        return py_save_path

    @abc.abstractmethod
    def _get_data(self, data_flows: list[np.ndarray]):
        raise NotImplementedError


class DatatransformerEven(DataTransformerBase):
    def __init__(self, file_path: str, max_flows: int, seq_len: int, label_len: int, pred_len: int, step_size=1,
                 aggregation_time=1000, processes=4):
        self.max_flows = max_flows
        self.seq_len = seq_len
        self.aggregation_time = aggregation_time
        self.processes = processes

        if label_len > seq_len:
            raise AttributeError("Label length has to be smaller then the sequence length")

        self.label_len = label_len
        self.pred_len = pred_len
        self.step_size = step_size

        super().__init__(file_path)

    def _get_data(self, data_flows: list[ndarray]):
        print("[+] Starting data preparation...")

        if self.seq_len is None or self.pred_len is None:
            raise AttributeError("Seq_len and pred_len have to be not None")

        # data_flows = self._list_milliseconds_only_sizes_(data_flows, self.aggregation_time)
        data_flows = split_list(data_flows, self.processes)

        params = [(data_flows[flow], self.seq_len, self.label_len,
                   self.pred_len, self.step_size, self.aggregation_time, flow) for flow in range(len(data_flows))]

        process_pool = multiprocessing.Pool(processes=self.processes)
        results = process_pool.starmap(self.__create_sequences__, params)
        process_pool.close()
        process_pool.join()

        flow_seq = []
        for res in results:
            flow_seq += res

        flows = {
            'shape': (self.seq_len, self.label_len, self.pred_len),
            'flow_seq': flow_seq,
        }

        print(f"[+] Found sequences: {sum([len(x) for x in flow_seq])} in {len(flow_seq)} flows sequences")
        return flows

    @staticmethod
    def __create_sequences__(data_flows: list[np.ndarray], seq_len: int, label_len: int, pred_len: int, step_size: int,
                             aggregation_time: int, flow_id: int):
        seq = []
        counter = 0

        for flow in data_flows:
            flow = _list_milliseconds_only_sizes_(data_flow=flow, aggregation_time=aggregation_time)

            flow_seq = []
            for i in range(0, len(flow) - seq_len - pred_len, step_size):  # len(flow)
                potential_seq = flow[i: i + seq_len + pred_len]
                zero_element = 0  # scaler.zero_element()

                if np.sum(potential_seq[:seq_len, 1] != zero_element) < (seq_len / 4) \
                        or np.sum(potential_seq[-pred_len:, 1] != zero_element) < (pred_len / 4):
                    continue

                flow_seq.append(potential_seq)

            if len(flow_seq) != 0:
                seq.append(flow_seq)
            counter += 1
            print(f"[+] {counter} / {len(data_flows)} | Found {len(flow_seq)} sequences | process id {flow_id}")
        return seq


class DataTransformerSinglePacketsEven(DataTransformerBase):
    def __init__(self, pcap_file_path: str, max_flows: int, seq_len: int, max_mil_seq: int, label_len: int,
                 pred_len: int, step_size=1, aggregation_time=1000, processes=4):
        self.max_flows = max_flows
        self.seq_len = seq_len
        self.max_mil_seq = max_mil_seq
        self.processes = processes
        self.aggregation_time = aggregation_time

        if label_len > seq_len:
            raise AttributeError("Label length has to be smaller then the sequence length")

        self.label_len = label_len
        self.pred_len = pred_len
        self.step_size = step_size

        super().__init__(pcap_file_path)

    def _get_data(self, data_flows: list[np.ndarray]):
        print("[+] Starting data preparation...")

        if self.seq_len is None or self.pred_len is None:
            raise AttributeError("Seq_len and pred_len have to be not None")

        # data_flows = self._list_milliseconds_(data_flows, self.aggregation_time)  # [ [ time, [packets] ] ]
        data_flows = split_list(data_flows, self.processes)

        params = [(data_flows[flow], self.seq_len, self.label_len,
                   self.pred_len, self.step_size, self.max_mil_seq, self.aggregation_time, flow) for flow in
                  range(len(data_flows))]

        process_pool = multiprocessing.Pool(processes=self.processes)
        results = process_pool.starmap(self.__create_sequences__, params)
        process_pool.close()
        process_pool.join()

        flow_seq_x, flow_seq_y = [], []
        for x, y in results:
            flow_seq_x += x
            flow_seq_y += y

        flows = {
            'shape': (self.seq_len, self.label_len, self.pred_len),
            'flow_seq': {'x': flow_seq_x, 'y': flow_seq_y},
        }

        print(f"[+] Found {sum(len(x) for x in flow_seq_x)} in {len(flow_seq_x)} flows sequences")
        return flows

    @staticmethod
    def __create_sequences__(data_flows: list, seq_len: int, label_len: int, pred_len: int, step_size: int,
                             max_mil_seq: int, aggregation_time: int, flow_id: int):
        seq_x = []
        seq_y = []

        counter = 0
        for flow in data_flows:
            flow = _list_milliseconds_(data_flow=flow, aggregation_time=aggregation_time)

            flow_seq_x = []
            flow_seq_y = []

            for i in range(max(label_len, max_mil_seq), len(flow),
                           step_size):  # at least label_len should be available - it might happen, that label_len is bigger then the sequence
                if i + pred_len > len(flow):  # not enough for prediction
                    break

                potential_seq = flow[i - max_mil_seq: i]
                potential_pred = flow[i - label_len: i + pred_len]

                # <<< x1|y1: (A <-> B) -> (A <-> B) >>>
                x1 = list(map(lambda z: z[1], potential_seq))
                x1 = list(itertools.chain(*x1))  # x = [len(x), 3] (time,size,direction,protocol)

                if len(x1) < seq_len:
                    continue

                x1 = np.stack(x1[-seq_len:])

                if sum(1 for element in x1 if element[1] != 0) < (seq_len / 4):
                    continue

                # create [time, size] vector - filter for (A -> B) packets
                y1 = list(map(lambda z: [z[0], sum(map(lambda t: t[1], z[1]))], potential_pred))  # y = [len(y), 1]

                if sum(1 for element in y1[label_len:] if element[1] != 0) < (pred_len / 4):
                    continue

                y1 = np.array(y1)

                flow_seq_x.append(x1)
                flow_seq_y.append(y1)

            if len(flow_seq_x) != 0:
                seq_x.append(flow_seq_x)
                seq_y.append(flow_seq_y)

            counter += 1
            print(f"[+] Process: {flow_id} | {counter} / {len(data_flows)}")

        return seq_x, seq_y


def visualize_flows(file_path: str):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    for flow in data:
        parsed_flow = _list_milliseconds_only_sizes_(flow, 1000)

        zeros = np.count_nonzero(parsed_flow[:, 1])
        values = len(parsed_flow)

        print(f"{zeros} / {values}")


def __create_split_flow_files__():
    path = 'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1\\univ1_pt1'  # _test
    path_test = 'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1\\univ1_pt1_test'  #

    save_path = 'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1\\univ1_pt1.pkl'
    save_path_test = 'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1\\univ1_pt1_test.pkl'

    parse_pcap_to_list(path, save_path)
    parse_pcap_to_list(path_test, save_path_test)


def __save_even__(pred_lens: list, load_path: str, aggr_time: list):
    for j in aggr_time:
        for i in pred_lens:
            save_path = f'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1\\univ1_pt1_even_336_48_{i}_{j}.pkl'
            data_transformer = DatatransformerEven(load_path, max_flows=-1, seq_len=336, label_len=48, pred_len=i,
                                                   step_size=1, aggregation_time=j, processes=8)

            data_transformer.save_python_object(save_path)
            print(f"[x] Finished {i} and saved it in {save_path}")

    print("<<<<<<<<<<<<<<<< Done Even >>>>>>>>>>>>>>>>")


def __save_single__(pred_lens: list, load_path: str, aggr_time: list):
    for j in aggr_time:
        for i in pred_lens:
            save_path = f'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1\\univ1_pt1_single_336_48_{i}_{j}.pkl'
            data_transformer = DataTransformerSinglePacketsEven(load_path, max_flows=-1, seq_len=336, label_len=48,
                                                                pred_len=i, max_mil_seq=500, step_size=1,
                                                                aggregation_time=j, processes=8)

            data_transformer.save_python_object(save_path)
            print(f"[x] Finished {i} and saved it in {save_path}")

    print("<<<<<<<<<<<<<<<< Done Single >>>>>>>>>>>>>>>>")


if __name__ == "__main__":
    print("<<<<<<<<<<<<<<<< Start >>>>>>>>>>>>>>>>")
    path = 'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1\\univ1_pt1.pkl'  # _test
    preds = [12, 18, 24, 30]
    aggregation_time = [1000, 100, 10, 1]  # 1000 = Milliseconds, 100 = 10xMilliseconds, 10 = 100xMilliseconds, 1 = Seconds

    visualize_flows(path)
    # __create_split_flow_files__() # only if packets got changed

    #__save_even__(preds, path, aggregation_time)
    #__save_single__(preds, path, aggregation_time)

    print("<<<<<<<<<<<<<<<< Done >>>>>>>>>>>>>>>>")
    sys.exit(0)
