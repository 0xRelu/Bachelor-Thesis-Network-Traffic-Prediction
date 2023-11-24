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
from numpy import ndarray

from utils.data_perperation import split_list, _list_milliseconds_only_sizes_, _list_milliseconds_only_sizes_not_np, \
    _split_flow_n, _list_milliseconds_, parse_pcap_to_list_n, \
    _split_flow_n2, create_test_from_full


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

    def _get_data(self, data_flows: list[ndarray]):
        print("[+] Starting data preparation...")

        # random.shuffle(data_flows)  # tries to balance load
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
    def __create_sequences__(data_flows: list[np.ndarray], aggregation_time: int, flow_id: int, min_length: int):
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
            flow = list(filter(lambda x: len(x) > min_length * 2, flow))
            # flow = list(map(mapping, flow))
            res_data_flows.extend(flow)

            if counter % 1000 == 0:
                print(f"[+] Found {len(res_data_flows)} in {counter / len(data_flows)} % | Splitted Flow Length Mean {mean([0] + [len(x) for x in res_data_flows])} | process id {flow_id}")
            counter += 1

        print(f"[+] Finally found {len(res_data_flows)} | Splitted Flow Length Mean {mean([0] + [len(x) for x in res_data_flows])} | process id {flow_id} | Loading timestamps....")

        for i in range(len(res_data_flows)):
            res_data_flows[i] = list(map(mapping, res_data_flows[i]))

        print(f"[++] Finished process with id {flow_id}")
        return res_data_flows


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
                   self.pred_len, self.step_size, self.max_mil_seq, self.aggregation_time, self.filter_size, flow) for
                  flow in
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

            if len(flow_packets) != len(flow_aggregated) or flow_packets[0][0] != flow_aggregated[0, 0] or \
                    flow_packets[-1][0] != flow_aggregated[-1, 0]:
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

            print(mean([t[1] for t in data['v_not_zero']]))

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
    print(f"Packets per flow: mean: {mean(p_count)}")

    # bytes per flow
    b_count = []
    for flow in data:
        b_count.append(sum([x[1] for x in flow]))
    print(f"Bytes per flow: {mean(b_count)}")

    # --- flow aggregated data
    start_points = []
    end_points = []
    agg_len = []
    v_not_zero = []
    corr = []
    hurst = []

    skipped = 0

    for flow in data:
        if len(flow) < 2:
            skipped += 1
            continue

        start_time = datetime.datetime.fromtimestamp(flow[0][0])
        end_time = datetime.datetime.fromtimestamp(flow[-1][0])
        duration = end_time - start_time

        duration = int(duration.total_seconds() * 1000)

        if duration == 0:
            skipped += 1
            continue

        # aggregated len per flow
        if duration > 3000000:
            start_counter = 0
            end_counter = 0
            for i in range(len(flow) - 3):
                if flow[i][-1] == 'FA':
                    end_counter += 1

                if flow[i][-1] == 'S' and \
                        flow[i + 1][-1] == 'SA' and \
                        flow[i + 2][-1] == 'A':
                    start_counter += 1

            start_points.append(start_counter)
            end_points.append(end_counter)
        agg_len.append(duration)

        unique_filled_milliseconds = set(map(lambda x: int(x[0] * 1000), flow))

        # Non zero values per flow
        v_not_zero.append((len(unique_filled_milliseconds), len(unique_filled_milliseconds) / duration))

        # Autocorrelation
        # [window_sizes_corr, corr_f] = calculate_autocorrelation(flow_ag[:, 1])
        # corr_f = sm.tsa.acf(flow_ag[:, 1], nlags=5)
        # corr.append(corr_f)

        # Hurst Exponent
        # H, c, [window_sizes_h, RS] = calculate_hurst_exponent(flow_ag[:, 1])
        # hurst.append(H)

        if len(v_not_zero) % 10000 == 0:
            print(f"[+] Finished {len(v_not_zero)} / {len(data)} : {len(v_not_zero) / len(data)}")

    print(f"Skipped about: {skipped}")
    print(f"Start counter mean: {mean(start_points)} | End Counter mean: {mean(end_points)}")
    print(f"Non zero values per flow:  mean: {mean([t[1] for t in v_not_zero])}")

    agg_len = sorted(agg_len)
    print(f"Time in milliseconds {agg_len[:5]}, {agg_len[-5:]}| (mean):{mean(agg_len)}")
    # print(f"Correlation per flow: {corr}")
    # print(f"Hurst per flow: {hurst}")

    data = {
        "f_count": f_count,
        "p_count": p_count,
        "b_count": b_count,
        "agg_len": agg_len,
        "v_not_zero": v_not_zero,
        # "hurst": hurst
    }

    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

    return data


def __create_split_flow_files__():
    path = 'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1\\univ1_pt'  # _test
    path = [path + str(i) for i in range(1, 21)]
    path_test = 'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1\\univ1_pt1_test'  #

    save_path = 'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1\\univ1_pt_n.pkl'
    save_path_test = 'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1\\univ1_pt1_test.pkl'

    parse_pcap_to_list_n(path, save_path)
    # parse_pcap_to_list_n([path_test], save_path_test)


def __save_even__(pred_lens: list, load_path: str, aggr_time: list):
    for j in aggr_time:
        for i in pred_lens:
            save_path = f'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1_p\\univ1_pt1_even_336_48_{i}_{j}.pkl'
            data_transformer = DatatransformerEven(load_path, max_flows=-1, seq_len=336, label_len=48, pred_len=i,
                                                   filter_size=100,
                                                   step_size=10, aggregation_time=j, processes=10)

            data_transformer.save_python_object(save_path)
            print(f"[x] Finished {i} and saved it in {save_path}")

    print("<<<<<<<<<<<<<<<< Done Even >>>>>>>>>>>>>>>>")


def __save_even_new2__(load_path: str, save_path: str, aggr_time: list):
    for j in aggr_time:
        save_path = save_path + f"_{j}.pkl"
        data_transformer = DatatransformerEvenSimple2(load_path, min_length=500, aggregation_time=j, processes=10)

        data_transformer.save_python_object(save_path)
        print(f"[x] Finished {j} and saved it in {save_path}")

    print("<<<<<<<<<<<<<<<< Done Even >>>>>>>>>>>>>>>>")


def __save_even_new__(load_path: str, aggr_time: list):
    for j in aggr_time:
        save_path = f'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1_n\\univ1_pt1_even2_{j}.pkl'
        data_transformer = DatatransformerEvenSimple(load_path, min_length=336, aggregation_time=j, processes=10)

        data_transformer.save_csv(save_path)
        print(f"[x] Finished {j} and saved it in {save_path}")

    print("<<<<<<<<<<<<<<<< Done Even >>>>>>>>>>>>>>>>")


def __save_single__(pred_lens: list, load_path: str, aggr_time: list):
    for j in aggr_time:
        for i in pred_lens:
            save_path = f'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1_p\\univ1_pt1_single_336_48_{i}_{j}.pkl'
            data_transformer = DataTransformerSinglePacketsEven(load_path, max_flows=-1, seq_len=336, label_len=48,
                                                                pred_len=i, max_mil_seq=100, step_size=1,
                                                                aggregation_time=j, filter_size=4, processes=10)

            data_transformer.save_python_object(save_path)
            print(f"[x] Finished {i} and saved it in {save_path}")

    print("<<<<<<<<<<<<<<<< Done Single >>>>>>>>>>>>>>>>")


if __name__ == "__main__":
    print("<<<<<<<<<<<<<<<< Start >>>>>>>>>>>>>>>>")
    path = 'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1\\univ1_pt_n.pkl'  # _test
    save_path = 'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1_n\\univ1_pt1_even3'

    test_path = 'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1\\univ1_pt_n_test.pkl'
    test_save_path = 'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1_n\\univ1_pt1_even3_test'

    save_analysis = 'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\ANALYSIS\\analysis_v2.pkl'  # _test
    preds = [12, 18, 24, 30]
    aggregation_time = [1000]  # 1000 = Milliseconds, 100 = 10xMilliseconds, 10 = 100xMilliseconds, 1 = Seconds

    # __create_split_flow_files__()
    # create_test_from_full(path, test_path, 2000, True)

    # _analyse_flows(path, save_analysis)
    # __create_split_flow_files__()  # only if packets got changed

    # __save_even_new__(path, aggregation_time)
    __save_even_new2__(test_path, test_save_path, aggregation_time)
    # __save_even__(preds, path, aggregation_time)
    # __save_single__(preds, path, aggregation_time)

    print("<<<<<<<<<<<<<<<< Done >>>>>>>>>>>>>>>>")
