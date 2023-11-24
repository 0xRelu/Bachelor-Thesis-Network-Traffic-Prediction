import math
import pickle
import random
import sys
from itertools import chain

import numpy as np
from scapy.layers.inet import TCP, IP, UDP
from scapy.packet import Packet
from scapy.plist import PacketList
from scapy.utils import rdpcap


def create_test_from_full(file_path: str, save_path: str, amount: int = 10, shuffle = False):
    print(f"[+] Loading flows from file with location: {file_path} ...")

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    if not shuffle:
        data = sorted(data, key=lambda x: len(x), reverse=True)
    else:
        random.shuffle(data)

    data = data[:amount]

    print(f"[+] Saving top {amount} flows in location: {save_path} ...")
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"[x] Saved ...")


def parse_pcap_to_list_n(paths: list, save_path: str):
    print(f"[+] Loading packets from pcap file with location: {paths} ...")

    size = 0

    for i in range(len(paths)):
        merge_counter = 0

        packetList = rdpcap(paths[i])
        ndata_flows = split_data_in_sockets(packetList)
        # ndata_flows = [split_in_flows(x) for x in ndata_flows.values()]
        # ndata_flows = list(chain(*ndata_flows))
        packetList = None

        if i > 0:
            with open(save_path, 'rb') as f:
                data_flows = pickle.load(f)

            for key, value in ndata_flows.items():
                if key in data_flows:
                    data_flows[key] += value
                    merge_counter += 1
                elif tuple(reversed(key)) in data_flows:
                    data_flows[tuple(reversed(key))] += value
                    merge_counter += 1
                else:
                    data_flows[key] = value
        else:
            data_flows = ndata_flows

        if i == len(paths) - 1:
            data_flows = [sorted(x, key=lambda t: t[0]) for x in data_flows.values()]
            # data_flows = [np.array(x) for x in data_flows]

        with open(save_path, 'wb') as f:
            pickle.dump(data_flows, f)

        size = len(data_flows)
        data_flows = None
        print(f"[x] Finished {paths[i]}. Found {size} flows total - merged {merge_counter}. Start loading next...")

    print(f"[+] Wrote {size} flows successfully in file with path {save_path}")


def split_in_flows(packets: list) -> list[list]:
    '''
    We expect TCP-Handshake -> Data -> TCP-Teardown
    # teardown
        if packets[i][-1] == 'FA' and \
                packets[i + 1][-1] == 'A' and \
                packets[i + 2][-1] == 'FA' and \
                packets[i + 3][-1] == 'A':
            if mode == 2:
                print("WTF")
            current += packets[i: i+4]
            i += 4
            mode = 2
    '''
    packets = sorted(packets, key=lambda x: x[0])
    tcpConst = 3

    flows = []
    current = []
    mode = 0  # 0 not defined, 1 after handshake, 2 after teardown

    i = 0
    arr_FA = [False, False]
    arr_A = [False, False]

    while i <= len(packets) - tcpConst:
        # teardown
        if packets[i][-1] == 'FA' and not arr_FA[0]:
            arr_FA[0] = True
        elif packets[i][-1] == 'A' and arr_FA[0] and not arr_A[0]:
            arr_A[0] = True
        elif packets[i][-1] == 'FA' and arr_FA[0] and arr_A[0] and not arr_FA[1]:
            arr_FA[1] = True
        elif packets[i][-1] == 'A' and arr_FA[1] and arr_A[0] and not arr_A[0]:
            arr_A[1] = True

        if all(arr_FA) and all(arr_A):
            mode = 2
            current.append(packets[i])
            flows.append(current)
            current = []

        # handshake
        if packets[i][-1] == 'S' and \
                packets[i + 1][-1] == 'SA' and \
                packets[i + 2][-1] == 'A':
            assert len(current) == 0 and (mode == 2 or mode == 0)
            current += packets[i: i + 3]
            i += 3  # jump to first new element
            mode = 1
            arr_FA = [False, False]
            arr_A = [False, False]
        else:
            if mode == 2:
                print("-------------------------------------------------------------FAILURE")
            current.append(packets[i])
            i += 1

    if len(current) != 0:
        flows.append(current)

    return flows


def split_data_in_sockets(packets: PacketList, max_flows: int = -1) -> dir:
    connection_map = {}

    counter = 0
    skip = 0

    for packet in packets:
        if TCP not in packet:
            skip += 1
            counter += 1
            if skip % 1000 == 0:
                print(f"Skipped {skip} / {counter}")
            continue

        forward_connection_id = (f"{packet[IP].src}|{packet[IP].sport}", f"{packet[IP].dst}|{packet[IP].dport}")

        if forward_connection_id in connection_map:
            connection_id = forward_connection_id
        elif tuple(reversed(forward_connection_id)) in connection_map:
            connection_id = tuple(reversed(forward_connection_id))
        else:
            connection_map[forward_connection_id] = []
            connection_id = forward_connection_id
            if len(connection_map) % 100 == 0:
                print(f"[+] \t Added new socket-to-socket connection: {len(connection_map)}")

        connection_map[connection_id].append(
            [float(packet.time), len(packet), 0 if connection_id == forward_connection_id else 1,
             hot_embed_protocol(packet), packet[TCP].flags])  #

        if counter % 5000 == 0:
            print(f"[+] Packets loaded: {counter / len(packets)}")
        counter += 1

    return connection_map


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


def _list_milliseconds_(data_flow: np.ndarray, aggregation_time: int = 1000) -> list:
    start_time = int(data_flow[0][0] * aggregation_time)
    end_time = int(data_flow[-1][0] * aggregation_time) + 1
    flow = [[time / aggregation_time, []] for time in range(start_time, end_time + 1)]

    for packet in data_flow:
        packet_time = int(packet[0] * aggregation_time)
        flow[packet_time - start_time][1].append(packet)

    return flow


def _list_milliseconds_only_sizes_(data_flow: np.ndarray, aggregation_time: int = 1000) -> np.ndarray:
    start_time = int(data_flow[0][0] * aggregation_time)  # assumes packets are ordered
    end_time = int(data_flow[-1][0] * aggregation_time) + 1
    flow = [[time / aggregation_time, 0] for time in range(start_time, end_time + 1)]

    for packet in data_flow:
        packet_time = int(packet[0] * aggregation_time)
        flow[packet_time - start_time][1] += packet[1]

    return np.array(flow)


def _list_milliseconds_only_sizes_not_np(data_flow: np.ndarray, aggregation_time: int = 1000) -> list:
    start_time = int(data_flow[0][0] * aggregation_time)  # assumes packets are ordered
    end_time = int(data_flow[-1][0] * aggregation_time) + 1
    flow = [[time / aggregation_time, 0] for time in range(start_time, end_time + 1)]

    for packet in data_flow:
        packet_time = int(packet[0] * aggregation_time)
        flow[packet_time - start_time][1] += packet[1]

    return flow


def _split_flow_n(split_flow: list[list], split_at: int, min_length: int, aggregation_time: int):
    current = []

    filtered_list = []
    zero_counter = 0

    # If 0 -> zero_counter + 1; current append 0
    # If 0 and zero_counter > x -> zero_counter += 1

    # If Z -> zero_counter = 0; non_zero_counter += 1; current append Z
    # If Z and zero_counter > x and non_zero_counter >= y -> filtered append Zeros and current; zero_counter = 0, non_zero_counter = 0
    # If Z; zero_counter > x; non_zero_counter < y -> zero_counter = 0; non_zero_counter = 0

    for value in split_flow:
        if value[1] == 0:
            zero_counter += 1
        else:
            if zero_counter > split_at and len(current) > (min_length + split_at):
                minZ = min(split_at, max(zero_counter - split_at, 0))
                filtered_list.extend(current)
                current = [[time / aggregation_time, 0] for time in
                           range(int(value[0] * aggregation_time) - minZ, int(value[0] * aggregation_time))]
            elif zero_counter > split_at:
                current = [[time / aggregation_time, 0] for time in
                           range(int(value[0] * aggregation_time) - split_at, int(value[0] * aggregation_time))]

            zero_counter = 0

        if zero_counter <= split_at:
            current.append(value)

    if len(filtered_list) > 2:
        before = [[time / aggregation_time, 0] for time in
                  range(int(filtered_list[0][0] * aggregation_time) - (split_at // 2),
                        int(filtered_list[0][0] * aggregation_time))]
        after = [[time / aggregation_time, 0] for time in range(int(filtered_list[-1][0] * aggregation_time) + 1,
                                                                int(filtered_list[-1][0] * aggregation_time) + (
                                                                        split_at // 2))]
        filtered_list = before + filtered_list + current + after if len(
            current) > min_length + split_at else before + filtered_list + after

    return filtered_list


def _split_flow_n2(split_flow: list[list], split_at: int):
    splitted_list = []
    current = []

    zero_counter = 0

    for value in split_flow:
        if value[1] == 0:
            zero_counter += 1
        else:
            zero_counter = 0

        if zero_counter <= split_at:
            current.append(value)
        elif zero_counter > split_at and len(current) != 0:
            splitted_list.append(current)
            current = []

    if len(current) != 0:
        splitted_list.append(current)

    return splitted_list


def calculate_hurst_exponent(data: np.ndarray):
    if len(data) < 10:
        return -1, -1, [[], []]

    segment_sizes = list(map(lambda x: int(10 ** x), np.arange(math.log10(10), math.log10(len(data)), 0.25))) + [
        len(data)]

    RS = []

    def _calc_rs(chunk):
        R = np.ptp(chunk)
        S = np.std(chunk)

        if R == 0 or S == 0:
            return 0

        return R / S

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


def split_by(tuples: list, percentages) -> list[dir]:
    segments = [int(len(list(chain(*tuples))) * per) for per in percentages]

    splits = []
    current = []
    current_len = 0

    for tuples in tuples:
        current.append(tuples)
        current_len += len(tuples)

        if current_len >= segments[len(splits)]:
            splits.append(current)
            current = []
            current_len = 0
            assert len(splits) <= len(segments)

    if len(current) != 0:
        splits.append(current)
    return splits


if __name__ == "__main__":

    seq = [[(0, i) for i in range(3)], [(2, i) for i in range(2)], [(4, i) for i in range(3)],
              [(5, i) for i in range(2)], [(6, i) for i in range(1)], [(7, i) for i in range(4)], [(8, i) for i in range(3)], [(9, i) for i in range(3)], [(10, i) for i in range(1)]]
    test = split_by(seq, [0.7, 0.2, 0.1])
    print(test)

    sys.exit(0)
    seq = [[0, 1], [0, 0], [0, 1], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 0], [0, 0], [0, 0],
           [0, 0], [0, 1]
        , [0, 1], [0, 0], [0, 0], [0, 0], [0, 1], [0, 1], [0, 0], [0, 0], [0, 1], [0, 1], [0, 0], [0, 0], [0, 0],
           [0, 0], [0, 0]]

    test = _split_flow_n2(seq, 3)
    print(test)

