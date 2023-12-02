import abc
import concurrent.futures
import csv
import itertools
import math
import multiprocessing
import os
import pickle

import datetime
import random
import sys
from statistics import mean

import numpy
import numpy as np
import torch
from numpy import ndarray

from utils.data_perperation import split_list, _list_milliseconds_only_sizes_, _list_milliseconds_only_sizes_not_np, \
    _split_flow_n, _list_milliseconds_, parse_pcap_to_list_n, \
    _split_flow_n2, create_test_from_full, _split_flow_tensor, _split_tensor_gpu


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

    def save_csv(self, csv_save_path: str):
        with open(csv_save_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(self.data)

        print(f"[+] Wrote {len(self.data)} successfully in file with path {csv_save_path}")

    @abc.abstractmethod
    def _get_data(self, data_flows: list[np.ndarray]):
        raise NotImplementedError


class DatatransformerEven(DataTransformerBase):
    def __init__(self, file_path: str, max_flows: int, seq_len: int, label_len: int, pred_len: int,
                 filter_size: int = 4,
                 step_size=1, aggregation_time: int = 1000, processes=4):
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

        data_flows = split_list(data_flows, self.processes)

        params = [(data_flows[flow], self.seq_len, self.label_len,
                   self.pred_len, self.step_size, self.aggregation_time, self.filter_size, flow) for flow in
                  range(len(data_flows))]

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


class DatatransformerEvenSimple(DataTransformerBase):
    def __init__(self, file_path: str, min_length: int = 100, aggregation_time: int = 1000, processes: int = 4):
        self.aggregation_time = aggregation_time
        self.processes = processes
        self.min_length = min_length

        super().__init__(file_path)

    def _get_data(self, data_flows: list[ndarray]):
        print("[+] Starting data preparation...")

        random.shuffle(data_flows)  # tries to balance load
        data_flows = split_list(data_flows, self.processes)

        print(f"[+] Starting {self.processes} processes containing: {[len(s) for s in data_flows]}")

        params = [(data_flows[flow], self.aggregation_time, flow, self.min_length) for flow in range(len(data_flows))]

        process_pool = multiprocessing.Pool(processes=self.processes)
        results = process_pool.starmap(self.__create_sequences__, params)
        process_pool.close()
        process_pool.join()

        flow_seq = []
        for res in results:
            flow_seq += res

        print(f"[+] Found sequences: {len(flow_seq)} in {sum(len(x) for x in data_flows)} flows sequences")

        print("[+] Parse timestamp...")
        flow_seq = list(
            map(lambda x: [datetime.datetime.fromtimestamp(x[0]).strftime('%Y-%m-%d %H:%M:%S.%f'), x[1]], flow_seq))

        flow_seq = [['date', 'bytes']] + flow_seq
        return flow_seq

    @staticmethod
    def __create_sequences__(data_flows: list[np.ndarray], aggregation_time: int, flow_id: int, min_length: int = 336):
        res_data_flows = []
        counter = 0
        skipped_counter = 0

        for flow in data_flows:
            flow = _list_milliseconds_only_sizes_not_np(data_flow=flow, aggregation_time=aggregation_time)  # aggregate

            if len(flow) < 8 * min_length:
                counter += 1
                skipped_counter += 1
                continue

            flow = _split_flow_n(split_flow=flow, split_at=min_length, min_length=8 * min_length,
                                 aggregation_time=aggregation_time)  # cleanup
            res_data_flows.extend(flow)

            if counter % 10 == 0:
                print(
                    f"[+] Found {len(res_data_flows)} in {counter / len(data_flows)} % | Skipped: {skipped_counter} | process id {flow_id}")
            counter += 1

        return res_data_flows


class DatatransformerEvenSimple2(DataTransformerBase):
    def __init__(self, file_path: str, min_length: int = 100, aggregation_time: int = 1000, processes: int = 4):
        self.aggregation_time = aggregation_time
        self.processes = processes
        self.min_length = min_length

        super().__init__(file_path)

    def _get_data(self, data_flows: dir):
        print("[+] Starting data preparation...")

        # random.shuffle(data_flows)  # tries to balance load
        data_flows = data_flows.values()
        data_flows = split_list(data_flows, self.processes)

        print(f"[+] Starting {self.processes} processes containing: {[len(s) for s in data_flows]}")

        params = [(data_flows[flow], self.aggregation_time, flow, self.min_length) for flow in range(len(data_flows))]

        process_pool = multiprocessing.Pool(processes=self.processes)
        results = process_pool.starmap(self.__create_sequences__, params)
        process_pool.close()
        process_pool.join()

        flow_seq = []
        for res in results:
            flow_seq += res

        print(f"[+] Found sequences: {len(flow_seq)} in {sum(len(x) for x in data_flows)} flows sequences")
        return flow_seq

    @staticmethod
    def __create_sequences__(data_flows: list[list], aggregation_time: int, flow_id: int, min_length: int):
        res_data_flows = []
        counter = 0

        def mapping(x):
            time = datetime.datetime.fromtimestamp(x[0])
            return [
                [time.month, time.day, time.weekday(), time.hour, time.minute, time.second, time.microsecond // 1000,
                 time.microsecond % 1000],
                x[1:]]

        for flow in data_flows:
            flow = _list_milliseconds_only_sizes_not_np(data_flow=flow,
                                                        aggregation_time=aggregation_time)  # returns [time in unix, bytes]
            flow = _split_flow_n2(split_flow=flow, split_at=min_length)
            flow = list(filter(lambda x: len(x) > min_length * 2, flow))  # filter all smaller than 1000 milliseconds
            # flow = list(map(mapping, flow))
            res_data_flows.extend(flow)

            if counter % 1000 == 0:
                print(
                    f"[+] Found {len(res_data_flows)} in {counter / len(data_flows)} % | Splitted Flow Length Mean {mean([0] + [len(x) for x in res_data_flows])} | process id {flow_id}")
            counter += 1

        print(
            f"[+] Finally found {len(res_data_flows)} | Splitted Flow Length Mean {mean([0] + [len(x) for x in res_data_flows])} | process id {flow_id} | Loading timestamps....")

        for i in range(len(res_data_flows)):
            res_data_flows[i] = list(map(mapping, res_data_flows[i]))

        print(f"[++] Finished process with id {flow_id}")
        return res_data_flows


class DatatransformerEvenSimpleGpu(DataTransformerBase):
    def __init__(self, file_path: str, min_length: int = 500, aggr: int = 1000):
        self.aggr = aggr
        self.min_length = min_length

        super().__init__(file_path)

    @staticmethod
    def _filter(data_flows: dict, aggr=1000, filter_tcp=True, filter_min_length=1000, get_max_Load=0.1, shuffle=12):
        keys = list(data_flows.keys())

        if filter_tcp:
            keys = [k for k in keys if k[0].startswith('TCP')]
            print(f"[+] Found {len(keys)} flows with filter tcp.")

        if filter_min_length is not None:
            keys = [k for k in keys if ((data_flows[k][-1][0] * aggr) - (data_flows[k][0][0] * aggr)) >= filter_min_length]

        if get_max_Load is not None:
            keys_load = [(k, 0 if ((data_flows[k][-1][0] * aggr) - (data_flows[k][0][0] * aggr)) < 1 else
                            sum([x[1] for x in data_flows[k]]) / ((data_flows[k][-1][0] * aggr) - (data_flows[k][0][0] * aggr))) for k
                                in keys]
            bef = sum([k[1] for k in keys_load])
            keys_load = sorted(keys_load, key=lambda x: x[1], reverse=True)
            keys_load = keys_load[:int(len(keys_load) * get_max_Load)]
            print(f"{sum([k[1] for k in keys_load]) / bef}")
            keys = [k[0] for k in keys_load]

        if shuffle is not None:
            random.seed(shuffle)
            random.shuffle(keys)

        return {key: data_flows[key] for key in keys}

    def _get_data(self, data_flows: dict):
        print("[+] Starting data preparation...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # random.shuffle(data_flows)  # tries to balance load
        data_flows = self._filter(data_flows, filter_tcp=True, filter_min_length=None, get_max_Load=None, shuffle=12)

        print(f"[+] Found {len(data_flows)} allowed flows after filtering.")

        data_flows = list(data_flows.values())

        for i in range(len(data_flows)):
            data_flows[i] = torch.tensor([x[:2] for x in data_flows[i]], device=device, dtype=torch.float64)

        flow_seq = []

        print("[+] Starting data flow transformation and splitting...")
        counter = 0
        for flow in data_flows:
            start_time = int(flow[0, 0] * self.aggr)  # assumes packets are ordered
            end_time = int(flow[-1, 0] * self.aggr) + 1
            flow_series_bytes = torch.zeros(end_time - start_time + 1, device=device, dtype=torch.float64)
            flow_series_time = torch.arange(start_time, end_time + 1, device=device, dtype=torch.float64)

            packet_times = ((flow[:, 0] * self.aggr) - start_time).long()
            flow_series_bytes.index_add_(0, packet_times, flow[:, 1])
            flow_series_bytes = torch.stack((flow_series_time / self.aggr, flow_series_bytes), dim=1)
            flow_series_bytes = _split_tensor_gpu(flow_series_bytes, consecutive_zeros=self.min_length)
            flow_series_bytes = [f.to('cpu') for f in flow_series_bytes if f.shape[0] > 2 * self.min_length]
            flow_seq.extend(flow_series_bytes)

            if counter % 1000 == 0:
                print(
                    f"[+] Found {len(flow_seq)} in {counter / len(data_flows)} % | Splitted Flow Length Mean {mean([0] + [len(x) for x in flow_seq])}")
            counter += 1

        print(
            f"[+] Found sequences: {len(flow_seq)} in {len(data_flows)} flows. Parsing timestamps....")

        def mapping2(time):
            return [time.month, time.day, time.weekday(), time.hour, time.minute, time.second, time.microsecond // 1000,
                    time.microsecond % 1000]

        def mapping(x):
            time = datetime.datetime.fromtimestamp(x[0, 0].item())
            time_in_tensor = [mapping2(time + datetime.timedelta(milliseconds=t_d)) for t_d in range(x.shape[0])]
            features = x[:, 1:].tolist()

            assert len(time_in_tensor) == len(features)

            combined = list(zip(time_in_tensor, features))
            combined = [list(pair) for pair in combined]
            return combined

        flow_seq = [mapping(x) for x in flow_seq]

        return flow_seq


def _create_split_flow_files():
    path = 'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1\\univ1_pt'  # _test
    path = [path + str(i) for i in range(1, 21)]
    path_test = 'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1\\univ1_pt1_test'  #

    save_path = 'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1\\univ1_pt_full_2.pkl'
    save_path_test = 'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1\\univ1_pt1_test.pkl'

    parse_pcap_to_list_n(path, save_path)
    # parse_pcap_to_list_n([path_test], save_path_test)cd


def _save_even_gpu(load_path: str, save_path: str, aggr_time: list):
    for j in aggr_time:
        save_path = save_path + f"_{j}.pkl"
        data_transformer = DatatransformerEvenSimpleGpu(load_path, min_length=500, aggr=j)

        data_transformer.save_python_object(save_path)
        print(f"[x] Finished aggr {j} and saved it in {save_path}")

    print("<<<<<<<<<<<<<<<< Done Even >>>>>>>>>>>>>>>>")


if __name__ == "__main__":
    print("<<<<<<<<<<<<<<<< Start >>>>>>>>>>>>>>>>")
    path = 'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1_n\\univ1_pt_full.pkl'  # univ1_pt_n
    save_path = 'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1_n\\univ1_pt1_even4'

    test_path = 'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1_n\\univ1_pt_n_test.pkl'
    test_save_path = 'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1_n\\univ1_pt1_even3_test'

    aggregation_time = [1000]  # 1000 = Milliseconds, 100 = 10xMilliseconds, 10 = 100xMilliseconds, 1 = Seconds

    # _create_split_flow_files()
    # create_test_from_full(path, test_path, 'TCP', 1000, True)

    # __save_even_new2__(test_path, test_save_path, aggregation_time)
    _save_even_gpu(path, save_path, aggr_time=aggregation_time)

    print("<<<<<<<<<<<<<<<< Done >>>>>>>>>>>>>>>>")
