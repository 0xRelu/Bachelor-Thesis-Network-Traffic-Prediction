import abc
import csv
import pickle

import datetime
import random
from statistics import mean

import numpy as np
import torch

from utils.data_preparation_tools import parse_pcap_to_list_n, split_tensor_gpu


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


class DatatransformerEvenSimpleGpu(DataTransformerBase):
    def __init__(self, file_path: str, consecutive_zeros=500, min_length: int = 1000, aggr: int = 1000):
        self.aggr = aggr
        self.min_length = min_length
        self.consecutive_zeros = consecutive_zeros

        super().__init__(file_path)

    @staticmethod
    def _filter(data_flows: dict, filter_tcp=True, shuffle=12):
        keys = list(data_flows.keys())

        if filter_tcp:
            keys = [k for k in keys if k[0].startswith('TCP')]
            print(f"[+] Found {len(keys)} flows with filter tcp.")

        # if filter_min_length is not None:
        #     keys = [k for k in keys if
        #             ((data_flows[k][-1][0] * aggr) - (data_flows[k][0][0] * aggr)) >= filter_min_length]
        #
        # if get_max_Load is not None:
        #     keys_load = [(k, 0 if ((data_flows[k][-1][0] * aggr) - (data_flows[k][0][0] * aggr)) < 1 else
        #     sum([x[1] for x in data_flows[k]]) / ((data_flows[k][-1][0] * aggr) - (data_flows[k][0][0] * aggr))) for k
        #                  in keys]
        #     bef = sum([k[1] for k in keys_load])
        #     keys_load = sorted(keys_load, key=lambda x: x[1], reverse=True)
        #     keys_load = keys_load[:int(len(keys_load) * get_max_Load)]
        #     print(f"{sum([k[1] for k in keys_load]) / bef}")
        #     keys = [k[0] for k in keys_load]

        if shuffle is not None:
            random.seed(shuffle)
            random.shuffle(keys)

        return {key: data_flows[key] for key in keys}

    def _get_data(self, data_flows: dict):
        print("[+] Starting data preparation...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # random.shuffle(data_flows)  # tries to balance load
        data_flows = self._filter(data_flows)

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

            if flow_series_bytes.shape[0] < self.min_length:
                counter += 1
                continue

            flow_series_time = torch.arange(start_time, end_time + 1, device=device, dtype=torch.float64)

            packet_times = ((flow[:, 0] * self.aggr) - start_time).long()
            flow_series_bytes.index_add_(0, packet_times, flow[:, 1])
            flow_series_bytes = torch.stack((flow_series_time / self.aggr, flow_series_bytes), dim=1)
            flow_series_bytes = split_tensor_gpu(flow_series_bytes, consecutive_zeros=self.consecutive_zeros)
            flow_series_bytes = [f.to('cpu') for f in flow_series_bytes if f.shape[0] > self.min_length]
            flow_seq.extend(flow_series_bytes)

            if counter % 1000 == 0:
                print(
                    f"[+] Found {len(flow_seq)} in {counter / len(data_flows)} % | Splitted Flow Length Mean {mean([0] + [len(x) for x in flow_seq])}")
            counter += 1

        print(f"[+] Found sequences: {len(flow_seq)} in {len(data_flows)} flows. "
              f"Real preds: {len([x for x in flow_seq if x.shape[0] > 2 * self.consecutive_zeros]) / len(flow_seq)} Parsing timestamps....")

        def mapping2(time):
            return [time.month, time.day, time.weekday(), time.hour, time.minute, time.second, time.microsecond // 1000,
                    time.microsecond % 1000]

        def mapping(x):
            time = datetime.datetime.fromtimestamp(x[0, 0].item())
            time_in_tensor = np.array([mapping2(time + datetime.timedelta(milliseconds=t_d)) for t_d in range(x.shape[0])])
            features = x[:, 1:].detach().numpy()

            return [time_in_tensor, features]

        counter = 0
        for i, x in enumerate(flow_seq):
            flow_seq[i] = mapping(x)

            if counter % 1000 == 0:
                print(f"Parsed {counter / len(flow_seq)}")
            counter += 1

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
        data_transformer = DatatransformerEvenSimpleGpu(load_path, consecutive_zeros=2500, min_length=3000, aggr=j)

        data_transformer.save_python_object(save_path)
        print(f"[x] Finished aggr {j} and saved it in {save_path}")

    print("<<<<<<<<<<<<<<<< Done Even >>>>>>>>>>>>>>>>")


if __name__ == "__main__":
    print("<<<<<<<<<<<<<<<< Start >>>>>>>>>>>>>>>>")
    path = 'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1_n\\univ1_pt_full.pkl'  # univ1_pt_n
    save_path = 'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1_n\\univ1_pt1_even4'

    test_path = 'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1_n\\univ1_pt_n_test.pkl'
    test_save_path = 'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1_n\\univ1_pt1_even4_long_test'

    filter_path = 'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1_n\\univ1_pt_filtered.pkl'
    filter_save_path = 'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1_n\\univ1_pt1_even4_long_filtered'

    aggregation_time = [1000]  # 1000 = Milliseconds, 100 = 10xMilliseconds, 10 = 100xMilliseconds, 1 = Seconds

    # _create_split_flow_files()
    # create_test_from_full(path, test_path, 'TCP', 1000, True)

    _save_even_gpu(filter_path, filter_save_path, aggr_time=aggregation_time)

    print("<<<<<<<<<<<<<<<< Done >>>>>>>>>>>>>>>>")
