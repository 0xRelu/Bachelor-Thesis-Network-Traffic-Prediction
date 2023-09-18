import abc
import concurrent.futures
import csv
import itertools
import multiprocessing
import os
import pickle
import sys

import numpy as np
import torch
from numpy import ndarray
from scapy.layers.inet import TCP, UDP, IP
from scapy.packet import Packet
from scapy.plist import PacketList
from scapy.utils import rdpcap
from torch import tensor, Tensor
from torch.nn.utils.rnn import pad_sequence

from utils.scaler import StandardScaler, StandardScalerNp, TrafficScalerLocalEvenNp, TrafficScalerGlobalEvenNp, \
    TrafficScalerLocalSingleNp


def _split_flows_(packets: PacketList, max_flows: int) -> list:
    data_flows = {}

    counter = 0

    for packet in packets:
        counter += 1

        if IP not in packet or TCP not in packet[IP]:  # only tcp for now
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
        data_flows.get(flow_id).append([packet_time, size, 1 if flow_id_1 is flow_id else 2])

        if counter % 5000 == 0:
            print(f"[+] Packets loaded: {counter / len(packets)}")

    return [np.array(flow) for flow in data_flows.values()]


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

    @staticmethod
    def _list_milliseconds_(data_flows: list[np.ndarray]) -> list:
        # Results add to list which contains all milliseconds

        res_data = []
        for value in data_flows:
            start_time = int(value[0][0] * 1000)
            end_time = int(value[-1][0] * 1000) + 1
            flow = [[time / 1000, []] for time in range(start_time, end_time + 1)]
            for packet in value:
                packet_time = int(packet[0] * 1000)
                flow[packet_time - start_time][1].append(packet)

            res_data.append(flow)

        return res_data


class DatatransformerEven(DataTransformerBase):
    def __init__(self, file_path: str, max_flows: int, seq_len: int, label_len: int, pred_len: int, step_size=1):
        self.max_flows = max_flows
        self.seq_len = seq_len

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

        data_flows = self._list_milliseconds_only_sizes_(data_flows)

        # normalize
        scaler = StandardScalerNp()

        seq = []
        processes = 4
        data_flows = split_list(data_flows, processes)

        params = [(data_flows[flow], self.seq_len, self.label_len,
                   self.pred_len, self.step_size, flow) for flow in range(len(data_flows))]

        process_pool = multiprocessing.Pool(processes=processes)
        results = process_pool.starmap(self.__create_sequences__, params)
        process_pool.close()
        process_pool.join()

        for result in results:
            seq += result

        seq = np.stack(seq)
        sizes = [int(0.8 * len(seq)), int(0.9 * len(seq))]

        train, test, vali = np.split(seq, sizes)
        train[:, 1] = scaler.fit_transform(train[:, 1])
        test[:, 1] = scaler.transform(test[:, 1])
        vali[:, 1] = scaler.transform(vali[:, 1])

        flows = {
            'shape': (self.seq_len, self.label_len, self.pred_len),
            'train': train,
            'test': test,
            'val': vali
        }

        print(f"[+] Found train: {len(train)}, test: {len(test)} and vali:{len(vali)}  sequences")
        return flows

    @staticmethod
    def _list_milliseconds_only_sizes_(data_flows: list[ndarray]) -> list[np.ndarray]:
        res_data = []
        for value in data_flows:
            start_time = int(value[0][0] * 1000)  # assumes packets are ordered
            end_time = int(value[-1][0] * 1000) + 1
            flow = [[time / 1000, 0] for time in range(start_time, end_time + 1)]

            for packet in value:
                packet_time = int(packet[0] * 1000)
                flow[packet_time - start_time][1] += packet[1]

            res_data.append(np.array(flow))

        return res_data

    @staticmethod
    def __create_sequences__(data_flows: list[np.ndarray], seq_len: int, label_len: int, pred_len: int, step_size: int,
                             flow_id: int):
        seq = []
        counter = 0

        for flow in data_flows:
            print(f"[+] Started process with id {flow_id} | {counter} / {len(data_flows)}")

            for i in range(0, len(flow) - seq_len - pred_len, step_size):
                potential_seq = flow[i: i + seq_len + pred_len]
                zero_element = 0  # scaler.zero_element()

                if np.sum(potential_seq[:seq_len, 1] == zero_element) > seq_len / 3 \
                        and np.sum(potential_seq[-pred_len:, 1] == zero_element) > pred_len / 3:
                    continue

                seq.append(potential_seq)

            counter += 1
        return seq


class DataTransformerSinglePacketsEven(DataTransformerBase):
    def __init__(self, pcap_file_path: str, max_flows: int, seq_len: int, max_mil_seq: int, label_len: int,
                 pred_len: int, step_size=1):
        self.max_flows = max_flows
        self.seq_len = seq_len
        self.max_mil_seq = max_mil_seq

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

        data_flows = self._list_milliseconds_(data_flows)  # [ [ time, [packets] ] ]

        seq_x = []
        seq_y = []

        processes = 4
        data_flows = split_list(data_flows, processes)

        params = [(data_flows[flow], self.seq_len, self.label_len,
                   self.pred_len, self.step_size, self.max_mil_seq, flow) for flow in range(len(data_flows))]

        process_pool = multiprocessing.Pool(processes=processes)
        results = process_pool.starmap(self.__create_sequences__, params)
        process_pool.close()
        process_pool.join()

        for x, y in results:
            seq_x += x
            seq_y += y

        seq_x = np.stack(seq_x)
        seq_y = np.stack(seq_y)

        sizes = [int(0.8 * len(seq_x)), int(0.9 * len(seq_x))]

        train_x, test_x, val_x = np.split(seq_x, sizes)
        train_y, test_y, val_y = np.split(seq_y, sizes)

        # scale output sizes
        scaler = StandardScalerNp()
        train_y[:, :, 1] = scaler.fit_transform(train_y[:, :, 1])
        test_y[:, :, 1] = scaler.transform(test_y[:, :, 1])
        val_y[:, :, 1] = scaler.transform(val_y[:, :, 1])

        flows = {
            'shape': (self.seq_len, self.label_len, self.pred_len),
            'train': {'x': train_x, 'y': train_y},
            'test': {'x': test_x, 'y': test_y},
            'val': {'x': val_x, 'y': val_y},
        }

        print(f"[+] Found Train: {len(flows['train']['x'])} | Test: {len(flows['test']['x'])} | Val: {len(flows['val']['x'])} sequences")
        return flows

    @staticmethod
    def __create_sequences__(data_flows: list, seq_len: int, label_len: int, pred_len: int, step_size: int,
                             max_mil_seq: int, flow_id: int):
        seq_x = []
        seq_y = []

        counter = 0
        for flow in data_flows:
            for i in range(max(label_len, max_mil_seq), len(flow),
                           step_size):  # at least label_len should be available - it might happen, that label_len is bigger then the sequence
                if i + pred_len > len(flow):  # not enough for prediction
                    break

                potential_seq = flow[i - max_mil_seq: i]
                potential_pred = flow[i - label_len: i + pred_len]

                # <<< x1|y1: (A <-> B) -> (A <-> B) >>>
                x1 = list(map(lambda z: z[1], potential_seq))
                x1 = list(itertools.chain(*x1))  # x = [len(x), 3] (time,size,direction)

                if len(x1) < seq_len:
                    continue

                x1 = np.stack(x1[-seq_len:])

                if sum(1 for element in x1 if element[1] != 0) <= len(x1) / 3:
                    continue

                # create [time, size] vector - filter for (A -> B) packets
                y1 = list(map(lambda z: [z[0], sum(map(lambda t: t[1], z[1]))], potential_pred))  # y = [len(y), 1]

                if sum(1 for element in y1[label_len:] if element[1] != 0) <= len(y1[label_len:]) / 3:
                    continue

                y1 = np.array(y1)

                if y1[:,
                   0].min() < 0:  # y1 can start (beacuse of label_length) before the input x1 -> then we have negative time values. Which we do not want!
                    continue

                seq_x.append(x1)
                seq_y.append(y1)

            print(f"[+] Process: {flow_id} | {counter} / {len(data_flows)}")
            counter += 1

        return seq_x, seq_y

def __save_even__(preds: list, path: str):
    for i in preds:
        save_path = f'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1\\univ1_pt1_even_336_48_{i}.pkl'
        data_transformer = DatatransformerEven(path, max_flows=-1, seq_len=336, label_len=48, pred_len=i, step_size=1)

        data_transformer.save_python_object(save_path)
        print(f"[x] Finished {i} and saved it in {save_path}")

    print("<<<<<<<<<<<<<<<< Done >>>>>>>>>>>>>>>>")


def __save_single__(preds: list, path: str):
    for i in preds:
        save_path = f'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1\\univ1_pt1_single_336_48_{i}.pkl'
        data_transformer = DataTransformerSinglePacketsEven(path, max_flows=-1, seq_len=336, label_len=48, pred_len=i,
                                                            max_mil_seq=500, step_size=1)

        data_transformer.save_python_object(save_path)
        print(f"[x] Finished {i} and saved it in {save_path}")

    print("<<<<<<<<<<<<<<<< Done >>>>>>>>>>>>>>>>")


if __name__ == "__main__":
    print("<<<<<<<<<<<<<<<< Start >>>>>>>>>>>>>>>>")
    path = 'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1\\univ1_pt1_test.pkl'  # _new.pcap
    preds = [12, 18, 24, 30]

    #__save_even__(preds, path)
    __save_single__(preds, path)

    print("<<<<<<<<<<<<<<<< Done >>>>>>>>>>>>>>>>")
    sys.exit(0)
