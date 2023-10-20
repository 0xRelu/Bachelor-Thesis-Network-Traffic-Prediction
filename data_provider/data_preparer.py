import abc
import concurrent.futures
import csv
import itertools
import math
import multiprocessing
import os
import pickle
import sys
import statsmodels.api as sm

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from scapy.layers.inet import TCP, UDP, IP, ICMP
from scapy.packet import Packet
from scapy.plist import PacketList
from scapy.utils import rdpcap

from utils.scaler import StandardScalerNp, LogScalerNp, RobustScalerNp, MinMaxScalerNp


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


def _split_flow(split_flow: np.ndarray, x: int, aggregation_time: int):
    # zero_indices = np.where(split_flow[:, 1] != 0)[0]
    # result = np.split(split_flow, zero_indices[np.where(np.diff(zero_indices) > x)[0] + 1])
    #
    # for i, res in enumerate(result):
    #     result[i] = result[i][:-np.where(res[:, 1] != 0)[0][-1]]
    splits = []
    current_split = []
    zero_counter = 0

    counter = 0

    def before_after(start, end) -> (list, list):
        return [np.array([time / aggregation_time, 0]) for time in range(start - int(x / 2), start)], \
            [np.array([time / aggregation_time, 0]) for time in range(aggregation_time, end + int(x / 2))]

    for value in split_flow:
        if value[1] == 0 and len(current_split) == 0:
            counter += 1
            continue

        if value[1] == 0:
            current_split.append(value)
            zero_counter += 1
            if zero_counter > x:
                before, after = before_after(int(current_split[0][0].item() * aggregation_time),
                                             int(current_split[-1][0].item() * aggregation_time))
                splits.append(np.stack(before + current_split + after))
                current_split = []
                zero_counter = 0
        else:
            current_split.append(value)
            zero_counter = 0

    if len(current_split) > 0:
        before, after = before_after(int(current_split[0][0].item() * aggregation_time),
                                     int(current_split[-1][0].item() * aggregation_time))
        splits.append(np.stack(before + current_split + after))

    return splits


def _split_flow_list(split_flow: list, x: int, aggregation_time: int):
    # zero_indices = np.where(split_flow[:, 1] != 0)[0]
    # result = np.split(split_flow, zero_indices[np.where(np.diff(zero_indices) > x)[0] + 1])
    #
    # for i, res in enumerate(result):
    #     result[i] = result[i][:-np.where(res[:, 1] != 0)[0][-1]]
    splits = []
    current_split = []
    zero_counter = 0

    counter = 0

    def before_after(start, end):
        return [[time / aggregation_time, []] for time in range(start - int(x / 2), start)], \
            [[time / aggregation_time, []] for time in range(end, end + int(x / 2))]

    for value in split_flow:
        if len(value[1]) == 0 and len(current_split) == 0:
            counter += 1
            continue

        if len(value[1]) == 0:
            current_split.append(value)
            zero_counter += 1
            if zero_counter > x:
                before, after = before_after(int(current_split[0][0] * aggregation_time),
                                             int(current_split[-1][0] * aggregation_time))
                splits.append(before + current_split + after)
                current_split = []
                zero_counter = 0
        else:
            current_split.append(value)
            zero_counter = 0

    if len(current_split) > 0:
        before, after = before_after(int(current_split[0][0] * aggregation_time),
                                     int(current_split[-1][0] * aggregation_time))
        splits.append(before + current_split + after)

    return splits


def calculate_hurst_exponent(data: np.ndarray):
    if len(data) < 10:
        return -1, -1, [[], []]

    segment_sizes = list(map(lambda x: int(10 ** x), np.arange(math.log10(10), math.log10(len(data)), 0.25))) + [len(data)]

    RS = []

    def _calc_rs(chunk):
        R = np.ptp(chunk)
        S = np.std(chunk)

        if R == 0 or S == 0:
            return 0

        return R/S

    for segment_size in segment_sizes:
        chunked_data = [data[i:i + segment_size] for i in range(0, len(data), segment_size)]
        w_rs = np.mean([_calc_rs(chunk) for chunk in chunked_data])
        RS.append(w_rs)

    A = np.vstack([np.log10(segment_sizes), np.ones(len(RS))]).T
    H, c = np.linalg.lstsq(A, np.log10(RS), rcond=-1)[0]

    return H, c, [segment_sizes, RS]


def calculate_autocorrelation(data: np.ndarray):
    segment_sizes = range(0, len(data), len(data) // 4)
    autocorrelation = []

    for lag in segment_sizes:
        data_shifted = np.roll(data, lag)

        centered_x = data - np.sum(data, axis=0, keepdims=True) / len(data)
        centered_y = data_shifted - np.sum(data_shifted, axis=0, keepdims=True) / len(data_shifted)
        cov_xy = 1. / (len(data) - 1) * np.dot(centered_x.T, centered_y)
        var_x = 1. / (len(data) - 1) * np.sum(centered_x ** 2, axis=0)
        var_y = 1. / (len(data_shifted) - 1) * np.sum(centered_y ** 2, axis=0)
        corrcoef_xy = cov_xy / np.sqrt(var_x[:, None] * var_y[None, :])

        autocorrelation.append(corrcoef_xy)

    return [segment_sizes, autocorrelation]


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
    def __init__(self, file_path: str, max_flows: int, seq_len: int, label_len: int, pred_len: int, filter_size:int = 4, step_size=1,
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
        self.filter_size = filter_size

        super().__init__(file_path)

    def _get_data(self, data_flows: list[ndarray]):
        print("[+] Starting data preparation...")

        if self.seq_len is None or self.pred_len is None:
            raise AttributeError("Seq_len and pred_len have to be not None")

        # data_flows = self._list_milliseconds_only_sizes_(data_flows, self.aggregation_time)
        data_flows = split_list(data_flows, self.processes)

        params = [(data_flows[flow], self.seq_len, self.label_len,
                   self.pred_len, self.step_size, self.aggregation_time, self.filter_size, flow) for flow in range(len(data_flows))]

        process_pool = multiprocessing.Pool(processes=self.processes)
        results = process_pool.starmap(self.__create_sequences__, params)
        process_pool.close()
        process_pool.join()

        flow_seq = []
        for res in results:
            flow_seq += res

        flows = {
            'shape': (self.seq_len, self.label_len, self.pred_len),
            'data': flow_seq,
        }

        print(f"[+] Found sequences: {sum([len(x[0]) for x in flow_seq])} in {len(flow_seq)} flows sequences")
        return flows

    @staticmethod
    def __create_sequences__(data_flows: list[np.ndarray], seq_len: int, label_len: int, pred_len: int, step_size: int,
                             aggregation_time: int, filter_size: int, flow_id: int):
        seq = []
        counter = 0

        for flow in data_flows:
            flow = _list_milliseconds_only_sizes_(data_flow=flow, aggregation_time=aggregation_time)  # aggregate

            only_sizes = flow[:, 1]
            without_zeros = only_sizes[np.nonzero(only_sizes)]
            mean = np.mean(without_zeros)
            var = np.var(without_zeros)
            length = len(without_zeros)

            flow_seq = []
            for i in range(0, len(flow) - seq_len - pred_len, step_size):  # len(flow)
                potential_seq = flow[i: i + seq_len + pred_len]
                zero_element = 0  # scaler.zero_element()

                if np.sum(potential_seq[:seq_len, 1] != zero_element) < filter_size:
                    continue

                flow_seq.append(potential_seq)

            if len(flow_seq) != 0:
                seq.append((flow_seq, mean, var, length))
            counter += 1
            print(f"[+] {counter} / {len(data_flows)} | Found {len(flow_seq)} sequences | process id {flow_id}")
        return seq


class DataTransformerSinglePacketsEven(DataTransformerBase):
    def __init__(self, pcap_file_path: str, max_flows: int, seq_len: int, max_mil_seq: int, label_len: int,
                 pred_len: int, filter_size: int, step_size=1, aggregation_time=1000, processes=4):
        self.max_flows = max_flows
        self.seq_len = seq_len
        self.max_mil_seq = max_mil_seq
        self.processes = processes
        self.aggregation_time = aggregation_time
        self.filter_size = filter_size

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

        data_flows = split_list(data_flows, self.processes)

        params = [(data_flows[flow], self.seq_len, self.label_len,
                   self.pred_len, self.step_size, self.max_mil_seq, self.aggregation_time, self.filter_size, flow) for flow in
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
            'data_x': flow_seq_x,
            'data_y': flow_seq_y,
        }

        print(f"[+] Found sequences: {sum([len(u[0]) for u in flow_seq_y])} in {len(flow_seq_x)} flows sequences")
        return flows

    @staticmethod
    def __create_sequences__(data_flows: list, seq_len: int, label_len: int, pred_len: int, step_size: int,
                             max_mil_seq: int, aggregation_time: int, filter_size: int, flow_id: int):
        seq_x = []
        seq_y = []

        counter = 0
        for flow in data_flows:
            flow_packets = _list_milliseconds_(data_flow=flow, aggregation_time=aggregation_time)
            flow_aggregated = _list_milliseconds_only_sizes_(data_flow=flow, aggregation_time=aggregation_time)
            # flow = _split_flow_list(split_flow=flow, x=max_mil_seq, aggregation_time=aggregation_time)

            if len(flow_packets) != len(flow_aggregated) or flow_packets[0][0] != flow_aggregated[0, 0] or flow_packets[-1][0] != flow_aggregated[-1, 0]:
                raise IndexError("Both aggregation types should have the same result!")

            only_sizes = flow_aggregated[:, 1]
            without_zeros = only_sizes[np.nonzero(only_sizes)]
            mean = np.mean(without_zeros)
            var = np.var(without_zeros)
            length = len(without_zeros)

            flow_seq_x = []
            flow_seq_y = []

            for i in range(max(label_len, max_mil_seq), len(flow_packets),
                           step_size):  # at least label_len should be available - it might happen, that label_len is bigger then the sequence
                if i + pred_len > len(flow_packets):  # not enough for prediction
                    break

                # <<< x1|y1: (A <-> B) -> (A <-> B) >>>
                x1 = flow_packets[i - max_mil_seq: i]
                x1 = list(map(lambda z: z[1], x1))
                x1 = list(itertools.chain(*x1))  # x = [len(x), 3] (time,size,direction,protocol)

                if len(x1) < seq_len:  # or sum(1 for element in x1 if element[1] != 0) < filter_size: not necessary
                    continue

                x1 = np.stack(x1[-seq_len:])

                # create [time, size] vector - filter for (A -> B) packets
                y1 = flow_aggregated[i - label_len: i + pred_len]

                flow_seq_x.append(x1)
                flow_seq_y.append(y1)

            if len(flow_seq_x) != 0:
                seq_x.append((flow_seq_x, mean, var, length))
                seq_y.append((flow_seq_y, mean, var, length))

            counter += 1
            print(f"[+] Process: {flow_id} | Sequences: {len(flow_seq_x)} | Flow {counter} / {len(data_flows)} "
                  f"| General: {sum([len(x[0]) for x in seq_x])}")
        return seq_x, seq_y


def _analyse_flows(file_path: str, save_path: str):
    if os.path.exists(save_path):
        try:
            with open(save_path, 'rb') as f:
                data = pickle.load(f)

            print(data)
            return
        except Exception:
            print("Could not pickle load data")
            return

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # --- flow general data

    # flow count
    f_count = len(data)
    print(f"Flow count: {f_count}")

    # packet count per flow
    p_count = [len(x) for x in data]
    print(f"Packets per flow: {p_count}")

    # bytes per flow
    b_count = []
    for flow in data:
        b_count.append(sum([x[1] for x in flow]))
    print(f"Bytes per flow: {b_count}")

    # --- flow aggregated data
    agg_len = []
    v_not_zero = []
    corr = []
    hurst = []

    for flow in data:
        flow_ag = _list_milliseconds_only_sizes_(data_flow=flow, aggregation_time=1000)

        # aggregated len per flow
        agg_len.append(len(flow_ag))

        # Non zero values per flow
        v_not_zero.append((len(np.nonzero(flow_ag[:, 1])[0]), len(np.nonzero(flow_ag[:, 1])[0]) / len(flow_ag[:, 1])))

        # Autocorrelation
        # [window_sizes_corr, corr_f] = calculate_autocorrelation(flow_ag[:, 1])
        # corr_f = sm.tsa.acf(flow_ag[:, 1], nlags=5)
        # corr.append(corr_f)

        # Hurst Exponent
        H, c, [window_sizes_h, RS] = calculate_hurst_exponent(flow_ag[:, 1])
        hurst.append(H)

        if len(v_not_zero) % 100 == 0:
            print(f"[+] Finished {len(v_not_zero)} / {len(data)} : {len(v_not_zero) / len(data)}")

    print(f"Non zero values per flow: {v_not_zero}")
    # print(f"Correlation per flow: {corr}")
    print(f"Hurst per flow: {hurst}")

    data = {
        "f_count": f_count,
        "p_count": p_count,
        "b_count": b_count,
        "agg_len": agg_len,
        "v_not_zero": v_not_zero,
        "hurst": hurst
    }

    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

    return data


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
            save_path = f'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1_p\\univ1_pt1_even_336_48_{i}_{j}.pkl'
            data_transformer = DatatransformerEven(load_path, max_flows=-1, seq_len=336, label_len=48, pred_len=i, filter_size=100,
                                                   step_size=1, aggregation_time=j, processes=8)

            data_transformer.save_python_object(save_path)
            print(f"[x] Finished {i} and saved it in {save_path}")

    print("<<<<<<<<<<<<<<<< Done Even >>>>>>>>>>>>>>>>")


def __save_single__(pred_lens: list, load_path: str, aggr_time: list):
    for j in aggr_time:
        for i in pred_lens:
            save_path = f'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1_p\\univ1_pt1_single_336_48_{i}_{j}.pkl'
            data_transformer = DataTransformerSinglePacketsEven(load_path, max_flows=-1, seq_len=336, label_len=48,
                                                                pred_len=i, max_mil_seq=100, step_size=1,
                                                                aggregation_time=j, filter_size=4, processes=8)

            data_transformer.save_python_object(save_path)
            print(f"[x] Finished {i} and saved it in {save_path}")

    print("<<<<<<<<<<<<<<<< Done Single >>>>>>>>>>>>>>>>")


if __name__ == "__main__":
    print("<<<<<<<<<<<<<<<< Start >>>>>>>>>>>>>>>>")
    path = 'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1\\univ1_pt1.pkl'  # _test
    save_analysis = 'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\ANALYSIS\\analysis_v1.pkl'  # _test
    preds = [12, 18, 24, 30]
    aggregation_time = [1000]  # 1000 = Milliseconds, 100 = 10xMilliseconds, 10 = 100xMilliseconds, 1 = Seconds

    # _analyse_flows(path, save_analysis)
    # __create_split_flow_files__() # only if packets got changed

    __save_even__(preds, path, aggregation_time)
    __save_single__(preds, path, aggregation_time)

    print("<<<<<<<<<<<<<<<< Done >>>>>>>>>>>>>>>>")
