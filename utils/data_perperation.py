import math
import pickle
import random
import re
import sys
from itertools import chain

import numpy as np
from scapy.layers.inet import TCP, IP, UDP
from scapy.packet import Packet, Raw
from scapy.plist import PacketList
from scapy.sendrecv import sniff
from scapy.sessions import TCPSession
from scapy.utils import rdpcap


def create_test_from_full(file_path: str, save_path: str, amount: int = 10, shuffle=False):
    print(f"[+] Loading flows from file with location: {file_path} ...")

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    if not shuffle:
        keys = sorted(data.keys(), key=lambda k: len(data[k]), reverse=True)
    else:
        keys = data.keys()
        random.shuffle(keys)

    keys = keys[:amount]
    res_data = {}

    for key in keys:
        res_data[key] = data[key]

    data = None
    print(f"[+] Saving top {amount} flows in location: {save_path} ...")
    with open(save_path, 'wb') as f:
        pickle.dump(res_data, f)

    print(f"[x] Saved ...")


def parse_pcap_to_list_n(paths: list, save_path: str):
    print(f"[+] Loading packets from pcap file with location: {paths} ...")

    size = 0

    for i in range(len(paths)):
        merge_counter = 0

        packetList = rdpcap(paths[i]).sessions()
        ndata_flows = split_data(packetList)
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
            for k, v in data_flows.items():
                data_flows[k] = sorted(v, key=lambda t: t[0])
            # data_flows = [np.array(x) for x in data_flows]

        with open(save_path, 'wb') as f:
            pickle.dump(data_flows, f)

        size = len(data_flows)
        data_flows = None
        print(f"[x] Finished {paths[i]}. Found {size} flows total - merged {merge_counter}. Start loading next...")

    print(f"[+] Wrote {size} flows successfully in file with path {save_path}")


def split_data(sessions: dir):
    flows = {}

    def parse_string(id_string):  # 'TCP 41.177.117.184:1618 > 41.177.3.224:51332'
        pattern = re.compile(
            r'(?P<protocol>\w+)\s(?P<src_ip>[\d.]+):(?P<src_port>\d+)\s>\s(?P<dst_ip>[\d.]+):(?P<dst_port>\d+).*')
        match = pattern.match(id_string)

        if match:
            protocol = match.group('protocol')
            src_ip = match.group('src_ip')
            src_port = match.group('src_port')
            dst_ip = match.group('dst_ip')
            dst_port = match.group('dst_port')
            return protocol, src_ip, src_port, dst_ip, dst_port
        else:
            # print(f"Could not find protocol, src or dst in: {id_string}")
            return None

    counter = 0

    protocols = set()

    for cid, packets in sessions.items():
        res = parse_string(cid)

        if res is None:
            continue
        else:
            protocol, sip, sport, dip, dport = res
            protocols.add(protocol)

        forward_connection_id = (f"{protocol}|{sip}|{sport}", f"{protocol}|{dip}|{dport}")

        if forward_connection_id in flows:
            connection_id = forward_connection_id
        elif tuple(reversed(forward_connection_id)) in flows:
            connection_id = tuple(reversed(forward_connection_id))
        else:
            flows[forward_connection_id] = []
            connection_id = forward_connection_id
            if len(flows) % 100 == 0:
                print(f"[+] \t Added new socket-to-socket connection: {len(flows)}")

        flows[connection_id].extend(list(map(lambda packet: [float(packet.time), len(packet),
                                                             0 if connection_id == forward_connection_id else 1,
                                                             protocol, "" if TCP not in packet else packet[TCP].flags], packets)))

        if counter % 5000 == 0:
            print(f"[+] Packets loaded: {counter / len(packets)}")
        counter += 1

    print(f"Found protocolls: {protocols}")
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


def _list_milliseconds_only_sizes_not_np(data_flow: list, aggregation_time: int = 1000) -> list:
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
           [(5, i) for i in range(2)], [(6, i) for i in range(1)], [(7, i) for i in range(4)],
           [(8, i) for i in range(3)], [(9, i) for i in range(3)], [(10, i) for i in range(1)]]
    test = split_by(seq, [0.7, 0.2, 0.1])
    print(test)

    sys.exit(0)
    seq = [[0, 1], [0, 0], [0, 1], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 0], [0, 0], [0, 0],
           [0, 0], [0, 1]
        , [0, 1], [0, 0], [0, 0], [0, 0], [0, 1], [0, 1], [0, 0], [0, 0], [0, 1], [0, 1], [0, 0], [0, 0], [0, 0],
           [0, 0], [0, 0]]

    test = _split_flow_n2(seq, 3)
    print(test)
