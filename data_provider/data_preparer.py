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


class DataTransformerBase:
    def __init__(self, pcap_file_path: str):
        self.file_path = pcap_file_path

        self.packets = self._load_packets()
        self.data = self._get_data(packets=self.packets)

    def _load_packets(self) -> PacketList:
        print(f"[+] Loading packets from pcap file with location: {self.file_path} ...")
        return rdpcap(self.file_path)

    @abc.abstractmethod
    def _get_data(self, packets: PacketList):
        raise NotImplementedError

    def save_csv(self, csv_save_path: str) -> str:
        with open(csv_save_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.data)

        print(f"[+] Wrote {len(self.data)} successfully in file with path {csv_save_path}")
        return csv_save_path

    def save_python_object(self, py_save_path: str) -> str:
        with open(py_save_path, 'wb') as f:
            pickle.dump(self.data, f)

        print(f"[+] Wrote {len(self.data)} successfully in file with path {py_save_path}")
        return py_save_path

    def save_tensor(self, tensor_save_path: str) -> str:
        directory = os.path.dirname(tensor_save_path)

        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(tensor_save_path, 'wb') as f:
            torch.save(self.data, f)

        print(f"[+] Wrote {len(self.data)} successfully in file with path {tensor_save_path}")
        return tensor_save_path

    @staticmethod
    def _get_protocol_data_length(packet: Packet) -> int:
        if TCP in packet:
            return 1

        if UDP in packet:
            return 2

        return 0

    @staticmethod
    def split_list(input_list, num_parts):
        avg = len(input_list) / float(num_parts)
        out = []
        last = 0.0

        while last < len(input_list):
            out.append(input_list[int(last):int(last + avg)])
            last += avg

        return out

    @staticmethod
    def _split_flows_(packets: PacketList, max_flows: int) -> list:
        data_flows = {}

        counter = 0

        for packet in packets:
            counter += 1

            if IP not in packet or TCP not in packet[IP]:
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

    @staticmethod
    def _list_milliseconds_only_sizes_(data_flows: list) -> list:
        res_data = []
        for value in data_flows:
            start_time = int(value[0][0] * 1000)
            end_time = int(value[-1][0] * 1000) + 1
            flow = np.zeros((end_time - start_time + 1, 1))

            for packet in value:
                packet_time = int(packet[0] * 1000)
                flow[packet_time - start_time, 0] += packet[1]

            res_data.append(flow)

        return res_data

    @staticmethod
    def normalize_packet_list(data_flows: dir) -> dir:
        ndata_flows = {}

        for key, value in data_flows.items():
            # scale sizes
            tensor = np.array(value, dtype=torch.float64)
            tensor_flow_mean = np.mean(tensor[:, 1])
            tensor_flow_std = np.std(tensor[:, 1])

            tensor[:, 1] = (tensor[:, 1] - tensor_flow_mean) / tensor_flow_std
            ndata_flows[key] = tensor

        return ndata_flows

    @staticmethod
    def _list_milliseconds_(data_flows: list[np.ndarray]) -> list:
        # Results add to list which contains all milliseconds

        res_data = []
        for value in data_flows:
            start_time = int(value[0][0] * 1000)
            end_time = int(value[-1][0] * 1000) + 1
            flow = [[time, []] for time in range(start_time, end_time)]
            for packet in value:
                packet_time = int(packet[0] * 1000)
                flow[packet_time - start_time][1].append(packet)

            res_data.append(flow)

        return res_data


class DatatransformerEven(DataTransformerBase):
    def __init__(self, pcap_file_path: str, max_flows: int, seq_len: int, label_len: int, pred_len: int, step_size=1):
        self.max_flows = max_flows
        self.seq_len = seq_len

        if label_len > seq_len:
            raise AttributeError("Label length has to be smaller then the sequence length")

        self.label_len = label_len
        self.pred_len = pred_len
        self.step_size = step_size

        super().__init__(pcap_file_path)

    def _get_data(self, packets: PacketList):
        print("[+] Attempting to save relevant data in list...")

        if self.seq_len is None or self.pred_len is None:
            raise AttributeError("Seq_len and pred_len have to be not None")

        data_flows = self._split_flows_(packets, self.max_flows)
        data_flows = self._list_milliseconds_only_sizes_(data_flows)

        # normalize
        scaler = TrafficScalerGlobalEvenNp()
        data_flows = scaler.fit_transform(data_flows)

        seq = []
        processes = 4
        data_flows = self.split_list(data_flows, processes)

        params = [(data_flows[flow], scaler, self.seq_len, self.label_len,
                   self.pred_len, self.step_size, flow) for flow in range(len(data_flows))]

        process_pool = multiprocessing.Pool(processes=processes)
        results = process_pool.starmap(self.__create_sequences__, params)
        process_pool.close()
        process_pool.join()

        for result in results:
            seq += result

        seq = np.stack(seq)

        permutation_indexes = np.random.permutation(len(seq))
        seq = seq[permutation_indexes]  # randomizes sequences

        flows = {
            'shape': (self.seq_len, self.label_len, self.pred_len),
            'seq': seq,
        }

        print(f"[+] Found {len(seq)} sequences")
        return flows

    @staticmethod
    def __create_sequences__(data_flows: list, scaler, seq_len: int, label_len: int, pred_len: int, step_size: int,
                             flow_id: int):
        seq = []
        counter = 0

        for flow in data_flows:
            print(f"[+] Started process with id {flow_id} | {counter} / {len(data_flows)}")

            for i in range(0, len(flow) - seq_len - pred_len, step_size):
                potential_seq = flow[i: i + seq_len + pred_len]
                zero_element = scaler.zero_element()

                if np.sum(potential_seq[:seq_len] == zero_element) > seq_len / 3 \
                        and np.sum(potential_seq[-pred_len:] == zero_element) > pred_len / 3:
                    continue

                seq.append(potential_seq)

            counter += 1
        return seq


# n = 0
# potential_points = [(a, b) for a in range(len(data_flows)) for b in range(data_flows[a].shape[0] - self.seq_len - self.pred_len)]  # TODO check
# potential_indices = list(range(len(potential_points)))
# res_sequences = np.zeros((self.batch_size, self.seq_len + self.pred_len, 1))
#
# print("[+] Starting to look for sequences")
#
# while n < self.batch_size:
#     if len(potential_indices) == 0:
#         res_sequences = res_sequences[n:, :, :]
#         print(f"[-] Only found {n / self.batch_size} of the sequences | {n} / {self.batch_size}")
#         break
#
#     if len(potential_indices) % 1000 == 0:
#         print(f"[+] Points left: {len(potential_indices)}")
#
#     rand_index = np.random.choice(potential_indices)
#     potential_indices.remove(rand_index)
#     (batch, start_point) = potential_points[rand_index]
#
#     potential_seq = data_flows[batch][start_point: start_point + self.seq_len + self.pred_len, :]
#     zero_element = scaler.zero_element(batch)  # TODO check if that makes sense
#
#     if np.sum(potential_seq[:self.seq_len] == zero_element) > self.seq_len / 3 \
#             and np.sum(potential_seq[-self.pred_len:] == zero_element) > self.pred_len / 3:
#         continue
#
#     res_sequences[batch, :, :] = potential_seq
#     n += 1
#
#     if n % 100 == 0:
#         print(f"[+] Found {n / self.batch_size} of the sequences | Points to check left: {len(potential_indices)}")
#
# return res_sequences


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

    def _get_data(self, packets: PacketList):
        print("[+] Attempting to save relevant data in list...")

        if self.seq_len is None or self.pred_len is None:
            raise AttributeError("Seq_len and pred_len have to be not None")
        scaler = TrafficScalerLocalSingleNp()

        data_flows = self._split_flows_(packets, self.max_flows)  # { flows }
        data_flows = scaler.fit_transform(data_flows)
        data_flows = self._list_milliseconds_(data_flows)  # [ [ time, [packets] ] ]

        seq_x = []
        seq_y = []

        processes = 4
        data_flows = self.split_list(data_flows, processes)

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

        permutation_indexes = np.random.permutation(len(seq_x))  # len(seq_x) == len(seq_y)
        seq_x = seq_x[permutation_indexes]  # randomizes sequences
        seq_y = seq_y[permutation_indexes]  # randomizes sequences

        flows = {
            'shape': (self.seq_len, self.label_len, self.pred_len),
            'x': seq_x,
            'y': seq_y
        }

        print(f"[+] Found {len(flows['x'])} sequences")
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
                min_time = x1[:, 0].min()
                x1[:, 0] = x1[:, 0] - min_time  # subtract min time value

                if sum(1 for element in x1 if element[1] != 0) <= len(x1) / 3:
                    continue

                # create [time, size] vector - filter for (A -> B) packets
                y1 = list(map(lambda z: [(z[0] / 1000) - min_time, sum(map(lambda t: t[1], z[1]))],
                              potential_pred))  # y = [len(y), 1]

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


class DataTransformerSinglePackets(DataTransformerBase):
    """
    DataTransformer for evenly spaced time series forecasting (zero values are not added yet -
    only later after creating the dataframe)
    Creates: [Flow, Time, Amount, Direction(0,1)]
    """

    def __init__(self, pcap_file_path: str, max_flows: int, seq_len: int, label_len: int, pred_len: int, step_size=1):
        self.max_flows = max_flows
        self.seq_len = seq_len

        if label_len > seq_len:
            raise AttributeError("Label length has to be smaller then the sequence length")

        self.label_len = label_len
        self.pred_len = pred_len
        self.step_size = step_size

        super().__init__(pcap_file_path)

    def _get_data(self, packets: PacketList):
        print("[+] Attempting to save relevant data in list...")

        if self.seq_len is None or self.pred_len is None:
            raise AttributeError("Seq_len and pred_len have to be not None")

        data_flows = self._split_flows_(packets, self.max_flows)
        flows = self._list_milliseconds_(data_flows)

        seq_x = []
        seq_y = []

        counter = 0
        for flow in flows:
            for i in range(0, len(flow), self.step_size):
                if i + self.seq_len + self.pred_len > len(flow):
                    break

                potential_seq = flow[i:i + self.seq_len]
                potential_pred = flow[i + self.seq_len - self.label_len: i + self.seq_len + self.pred_len]

                # skip if not at least one element not 0
                if sum(1 for element in potential_seq if not element[1]) <= len(potential_seq) / 2 \
                        or sum(1 for element in potential_pred if not element[1]) <= len(potential_pred) / 2:
                    continue  # TODO Check

                # <<< x1|y1: (A <-> B) -> (A <-> B) >>>
                x1 = list(map(lambda z: z[1], potential_seq))
                x1 = list(itertools.chain(*x1))  # x = [len(x), 3] (time,size,direction)
                x1 = list(map(lambda t: [t[0] - x1[0][0]] + t[1:], x1))  # subtract start time value
                x1 = torch.tensor(x1)

                # create [time, size] vector - filter for (A -> B) packets
                y1 = list(map(lambda z: [z[0], sum(map(lambda t: t[1], z[1]))], potential_pred))  # y = [len(y), 1]
                y1 = torch.tensor(y1)

                seq_x.append(x1)
                seq_y.append(y1)

            print(f"Flow {counter} / {len(flows)}")
            counter += 1

        # seq_y = torch.stack(seq_y, dim=0).to(torch.float64)
        # seq_y[:, :, 1] = (seq_y[:, :, 1] - torch.min(seq_y[:, :, 1])) / (torch.max(seq_y[:, :, 1]) - torch.min(seq_y[:, :, 1]))  # Min-Max normalization
        # seq_y = torch.unbind(seq_y, dim=0)  # transform tensor back to list of tensors to adapt to dataloader

        # scale data
        # total_mean = sum(tensor.mean(dim=(0, 2)) for tensor in tensors_list) / len(tensors_list)
        # total_std = sum(tensor.std(dim=(0, 2)) for tensor in tensors_list) / len(tensors_list)

        flows = {
            'shape': (self.seq_len, self.label_len, self.pred_len),
            'x': seq_x,
            'y': seq_y
        }

        print(f"[+] Found {len(flows['x'])} sequences")
        return flows


@DeprecationWarning
class DataTransformerSinglePacketsSplit(DataTransformerBase):
    """
    DataTransformer for evenly spaced time series forecasting (zero values are not added yet -
    only later after creating the dataframe)
    Creates: [Flow, Time, Amount, Direction(0,1)]
    """

    def __init__(self, pcap_file_path: str, max_flows: int, seq_len: int = None, pred_len: int = None, step_size=1):
        self.max_flows = max_flows
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.step_size = step_size

        super().__init__(pcap_file_path)

    def _get_data(self, packets: PacketList):
        print("[+] Attempting to save relevant data in list...")

        if self.seq_len is None or self.pred_len is None:
            raise AttributeError("Seq_len and pred_len have to be not None")

        data_flows = self._split_flows_(packets, self.max_flows)
        flows = self._list_milliseconds_(data_flows)

        seq_x = []
        seq_y = []

        counter = 0
        for flow in flows:
            for i in range(0, len(flow), self.step_size):
                if i + self.seq_len + self.pred_len > len(flow):
                    break

                potential_seq = flow[i:i + self.seq_len]
                potential_pred = flow[i + self.seq_len: i + self.seq_len + self.pred_len]

                # skip if not at least one element not 0
                if all(not entry[1] for entry in potential_seq) or all(not entry[1] for entry in potential_pred):
                    continue

                # <<< x1|y1: (A <-> B) -> (A -> B) >>>
                x1 = list(map(lambda z: z[1], potential_seq))
                x1 = list(itertools.chain(*x1))  # x = [len(x), 3] (time,size,direction)
                x1 = list(map(lambda t: [t[0] - x1[0][0]] + t[1:], x1))  # subtract start time value

                # create [time, size] vector - filter for (A -> B) packets
                y1 = list(map(lambda z: [z[0], sum(map(lambda t: t[1], list(filter(lambda u: u[2] == 1, z[1])))), 1],
                              potential_pred))  # y = [len(y), 1]

                seq_x.append(torch.tensor(x1).to_sparse_coo())
                seq_y.append(torch.tensor(y1).to_sparse_coo())

                # <<< x2|y2: (B <-> A) -> (A <- B) >>>
                x2 = list(map(lambda z: z[1], potential_pred))
                x2 = list(itertools.chain(*x2))  # x = [len(x), 3] (time,size,direction)
                x2 = list(map(lambda t: [t[0] - x2[0][0]] + t[1:], x2))  # subtract start time value

                # revert direction of packets
                x2 = list(map(lambda k: k[:-1] + [1 if k[-1] == 2 else 2], x2))

                # create [time, size] vector - filter for (A <- B) packets
                y2 = list(map(lambda z: [z[0], sum(map(lambda t: t[1], list(filter(lambda u: u[2] == 2, z[1])))), 2],
                              potential_pred))  # y = [len(y), 1]
                # (A <-> B) -> ( A <-> B)  <6200 - 6400>

                seq_x.append(torch.tensor(x2))
                seq_y.append(torch.tensor(y2))

            print(f"Flow {counter} / {len(flows)}")
            counter += 1

        flows = {
            'x': seq_x,
            'y': seq_y
        }

        print(f"[+] Found {len(flows['x'])} sequences")
        return flows


def __save_even__(preds: list, path: str):
    for i in preds:
        save_path = f'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1\\univ1_pt1_even_336_48_{i}.pkl'
        data_transformer = DatatransformerEven(path, max_flows=-1, seq_len=336, label_len=48, pred_len=i, step_size=1)

        data_transformer.save_tensor(save_path)
        print(f"[x] Finished {i} and saved it in {save_path}")

    print("<<<<<<<<<<<<<<<< Done >>>>>>>>>>>>>>>>")


def __save_single__(preds: list, path: str):
    for i in preds:
        save_path = f'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1\\univ1_pt1_single_336_48_{i}.pkl'
        data_transformer = DataTransformerSinglePacketsEven(path, max_flows=-1, seq_len=336, label_len=48, pred_len=i,
                                                            max_mil_seq=500, step_size=1)

        data_transformer.save_tensor(save_path)
        print(f"[x] Finished {i} and saved it in {save_path}")

    print("<<<<<<<<<<<<<<<< Done >>>>>>>>>>>>>>>>")


if __name__ == "__main__":
    print("<<<<<<<<<<<<<<<< Start >>>>>>>>>>>>>>>>")
    path = 'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1\\univ1_pt1'  # _new.pcap
    preds = [ 96]

    # __save_even__(preds, path)
    __save_single__(preds, path)

    print("<<<<<<<<<<<<<<<< Done >>>>>>>>>>>>>>>>")
    sys.exit(0)
