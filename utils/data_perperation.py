import math
import pickle
import random
import re
import subprocess
import sys
from itertools import chain

import numpy as np
import torch
from scapy.layers.http import HTTP
from scapy.layers.inet import TCP, IP, UDP
from scapy.packet import Packet, Raw
from scapy.plist import PacketList
from scapy.sendrecv import sniff
from scapy.sessions import TCPSession
from scapy.utils import rdpcap
from torch import tensor


def create_test_from_full(file_path: str, save_path: str, amount: int = 10, shuffle=False):
    print(f"[+] Loading flows from file with location: {file_path} ...")

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    if not shuffle:
        keys = sorted(list(data.keys()), key=lambda k: len(data[k]), reverse=True)
    else:
        keys = list(data.keys())
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

            # for key, value in ndata_flows.items():
            #     if key in data_flows:
            #         data_flows[key] += value
            #         merge_counter += 1
            #     elif tuple(reversed(key)) in data_flows:
            #         data_flows[tuple(reversed(key))] += value
            #         merge_counter += 1
            #     else:
            #         data_flows[key] = value

            for key, value in ndata_flows.items():
                if key in data_flows or tuple(reversed(key)) in data_flows:
                    data_flows[(key[0] + "-" + str(merge_counter), key[1] + "-" + str(merge_counter))] = value
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
                                                             protocol, "" if TCP not in packet else packet[TCP].flags, "" if HTTP not in packet else packet[HTTP].summary()],
                                             packets)))

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


def analyze_flows(files: list):
    for file_ in files:
        command_ = f"tshark -r {file_} -qz conv,tcp"
        output_ = subprocess.check_output(command_, shell=True, text=True)
        analyze_flow(output_)


def analyze_flow(output):
    flows = re.findall(r'\d+\.\d+\.\d+\.\d+:\d+ -> \d+\.\d+\.\d+\.\d+:\d+', output)

    for flow in flows:
        print("Flow:", flow)
        flow_info = subprocess.check_output(f"tshark -r your_file.pcap -qz conv,tcp -z follow,tcp,ascii,{flow}",
                                            shell=True)

        packet_count = int(re.search(r'Number of packets: (\d+)', flow_info).group(1))
        total_bytes = int(re.search(r'Length of this packet \(\): (\d+)', flow_info).group(1))

        print("Number of Packets:", packet_count)
        print("Total Bytes:", total_bytes)

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


def split_dict(dictionary, num_parts):
    dict_length = len(dictionary)
    items_per_part = math.ceil(dict_length / num_parts)

    split_dicts = []
    current_part = {}

    for idx, (key, value) in enumerate(dictionary.items()):
        current_part[key] = value

        if (idx + 1) % items_per_part == 0 or (idx + 1) == dict_length:
            split_dicts.append(current_part)
            current_part = {}

    return split_dicts


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


def _list_milliseconds_only_sizes_torch(data_flow: tensor, device, aggregation_time: int = 1000) -> tensor:
    start_time = int(data_flow[0, 0] * aggregation_time)  # assumes packets are ordered
    end_time = int(data_flow[-1, 0] * aggregation_time) + 1
    flow = torch.zeros(end_time - start_time + 1, device=device)

    packet_times = (data_flow[:, 0] * aggregation_time).int()
    flow[packet_times - start_time] += data_flow[:, 1]

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


def _split_flow_tensor(split_flow: tensor, consecutive_zeros: int):
    zero_indices = torch.where(split_flow[:, 1] == 0)[0]
    split_indices = []
    rem = []

    current_count = 0

    for i in range(1, len(zero_indices)):
        if zero_indices[i] == zero_indices[i - 1] + 1:
            current_count += 1
        else:
            if current_count >= consecutive_zeros:
                split_indices.append(zero_indices[i - 1].int().item() + 1)
                rem.append(len(split_indices) - 1)
            current_count = 0

        if current_count == consecutive_zeros:
            split_indices.append(zero_indices[i].int().item())

    if zero_indices[-1] < len(split_flow) - 1:
        split_indices.append(zero_indices[-1].int().item() + 1)
        rem.append(len(split_indices) - 1)

    # split_indices.append(len(split_flow))
    splitted_list = torch.tensor_split(split_flow, split_indices, dim=0)

    if torch.all(splitted_list[-1][:, 1] == 0).item():
        rem.append(len(split_indices))

    splitted_list = [t for i, t in enumerate(splitted_list) if i not in rem]

    return splitted_list


def _split_tensor_gpu(split_flow, consecutive_zeros):
    zero_indices = torch.nonzero(split_flow[:, 1] == 0).view(-1)

    if len(zero_indices) == 0:
        return [split_flow]

    splitted_list = []
    first_index = 0
    zero_counter = 0

    for i in range(1, len(zero_indices)):
        if zero_indices[i] - zero_indices[i - 1] == 1:
            zero_counter += 1
        else:
            zero_counter = 0

        if zero_counter == consecutive_zeros:
            splitted_list.append(split_flow[first_index:zero_indices[i]])
            first_index = zero_indices[i] + 1

        if zero_counter > consecutive_zeros:
            first_index = zero_indices[i] + 1

    if first_index <= len(split_flow) - 1:
        splitted_list.append(split_flow[first_index:])

    return splitted_list


def calculate_hurst_exponent(data: torch, device):
    if len(data) < 10:
        return -1, -1, [[], []]

    segment_sizes = torch.tensor([int(10 ** x) for x in torch.arange(math.log10(10), math.log10(len(data)), 0.25)]
                                 + [len(data)], device=device)
    RS = []

    def _calc_rs(chunk):
        R = torch.max(chunk) - torch.min(chunk)
        S = torch.std(chunk)

        if R == 0 or S == 0:
            return torch.tensor(0.0, device=device, dtype=torch.float64)

        return R / S

    for segment_size in segment_sizes:
        chunked_data = [data[i:i + segment_size] for i in range(0, len(data), segment_size)]
        w_rs = torch.mean(torch.stack([_calc_rs(chunk) for chunk in chunked_data]))
        RS.append(w_rs)

    A = torch.vstack([torch.log10(segment_sizes), torch.ones(len(RS), device=device, dtype=torch.float64)]).T
    H, c = torch.linalg.lstsq(A, torch.log10(torch.tensor(RS, device=device, dtype=torch.float64)), rcond=-1)[0]

    return H.item(), c.item(), [segment_sizes, RS]


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


def test_split_tensor(func):
    def are_lists_of_tensors_equal(list1, list2):
        if len(list1) != len(list2):
            return False

        for tensor1, tensor2 in zip(list1, list2):
            if not torch.equal(tensor1, tensor2):
                return False

        return True

    seq = torch.tensor([[0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 1], [0, 1]])
    res = [torch.tensor([[0, 1], [0, 0], [0, 0]]), torch.tensor([[0, 1], [0, 1], [0, 1]])]

    seq = func(seq, 2)
    assert are_lists_of_tensors_equal(seq, res)

    seq = torch.tensor([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
    res = [torch.tensor([[0, 0], [0, 0]])]

    seq = func(seq, 2)
    assert are_lists_of_tensors_equal(seq, res)

    seq = torch.tensor([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 1]])
    res = [torch.tensor([[0, 0], [0, 0]]), torch.tensor([[0, 1]])]

    seq = func(seq, 2)
    assert are_lists_of_tensors_equal(seq, res)

    seq = torch.tensor([[0, 1], [0, 0], [0, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0]])
    res = [torch.tensor([[0, 1], [0, 0], [0, 0], [0, 1], [0, 0], [0, 0]])]

    seq = func(seq, 2)
    assert are_lists_of_tensors_equal(seq, res)

    seq = torch.tensor([[0, 1], [0, 0], [0, 1], [0, 1], [0, 0], [0, 0], [0, 0], [0, 1], [0, 0], [0, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
    res = [torch.tensor([[0, 1], [0, 0], [0, 1], [0, 1], [0, 0], [0, 0]]),
           torch.tensor([[0, 1], [0, 0], [0, 0], [0, 1], [0, 0], [0, 0]])]

    seq = func(seq, 2)
    assert are_lists_of_tensors_equal(seq, res)

    seq = torch.tensor([[0, 1], [0, 0], [0, 1], [0, 1], [0, 0], [0, 0], [0, 0], [0, 1], [0, 0], [0, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
    res = [torch.tensor([[0, 1], [0, 0], [0, 1], [0, 1], [0, 0], [0, 0], [0, 0], [0, 1], [0, 0], [0, 0], [0, 1], [0, 0], [0, 0], [0, 0]])]

    seq = func(seq, 3)
    assert are_lists_of_tensors_equal(seq, res)


    print("Success")
    sys.exit(0)


if __name__ == "__main__":
    test_split_tensor(_split_tensor_gpu)

    seq = [[(0, i) for i in range(3)], [(2, i) for i in range(2)], [(4, i) for i in range(3)],
           [(5, i) for i in range(2)], [(6, i) for i in range(1)], [(7, i) for i in range(4)],
           [(8, i) for i in range(3)], [(9, i) for i in range(3)], [(10, i) for i in range(1)]]
    test = split_by(seq, [0.7, 0.2, 0.1])
    print(test)

