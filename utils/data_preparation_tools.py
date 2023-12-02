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
from scapy.packet import Packet
from scapy.plist import PacketList
from scapy.utils import rdpcap
from torch import tensor


def create_test_from_full(file_path: str, save_path: str, filter_prot: str = 'TCP', amount: int = 10,
                          shuffle=False):  # options HTTP, TPC, None=Nothing
    print(f"[+] Loading flows from file with location: {file_path} ...")

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    keys = list(data.keys())

    if filter_prot == 'TCP':
        keys = [k for k in keys if k[0].startswith('TCP')]
    elif filter_prot == 'HTTP':
        keys = [k for k in keys if
                any(packet[3] == "HTTP" for packet in data[k])]  # check if one packet has http layer in the flow
        print(len(keys))

    print(f"[+] Found {len(keys)} flows with filter {filter_prot}.")

    if not shuffle:
        keys = sorted(keys, key=lambda k: sum([p[1] for p in data[k]]), reverse=True)
    else:
        random.shuffle(keys)

    keys = keys[:amount]
    res_data = {}

    for key in keys:
        res_data[key] = data[key]

    data = None
    print(f"[+] Saving top {len(res_data)} flows in location: {save_path} ...")
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

            # for key, value in ndata_flows.items():
            #     if key in data_flows or tuple(reversed(key)) in data_flows:
            #         data_flows[(key[0] + "-" + str(merge_counter), key[1] + "-" + str(merge_counter))] = value
            #         merge_counter += 1
            #     else:
            #         data_flows[key] = value

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
                                                             protocol if HTTP not in packet else 'HTTP',
                                                             "" if TCP not in packet else packet[TCP].flags],
                                             packets)))

        if counter % 5000 == 0:
            print(f"[+] Packets loaded: {counter / len(packets)}")
        counter += 1

    print(f"Found protocolls: {protocols}")
    return flows


def split_tensor_gpu(tensor_, consecutive_zeros):
    # step 1: identify Zero Sequences
    # create a mask of zeros and find the difference between consecutive elements
    is_zero = tensor_[:, 1] == 0
    diff = torch.diff(is_zero.float(), prepend=torch.tensor([0.0], device=tensor_.device))

    # start and end indices of zero sequences
    start_indices = torch.where(diff == 1)[0]
    end_indices = torch.where(diff == -1)[0]

    # adjust for cases where sequences reach the end of the tensor
    if len(end_indices) == 0 or (len(start_indices) > 0 and end_indices[-1] < start_indices[-1]):
        end_indices = torch.cat([end_indices, tensor_.size(0) * torch.ones(1, dtype=torch.long, device=tensor_.device)])

    # step 2: mark split points
    # find sequences with length >= consecutive_zeros
    valid_seqs = (end_indices - start_indices) > consecutive_zeros
    valid_start_indices = start_indices[valid_seqs] + consecutive_zeros  # 0:st+2
    valid_end_indices = end_indices[valid_seqs]

    splits = []
    end_idx = 0
    for i in range(len(valid_start_indices)):
        splits.append(tensor_[end_idx:valid_start_indices[i]])
        end_idx = valid_end_indices[i]

    # add the remaining part of the tensor if any
    if end_idx < tensor_.size(0):
        splits.append(tensor_[end_idx:])

    return splits


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

    seq = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 0], [0, 0], [0, 0], [0, 1], [0, 1]])
    res = [torch.tensor([[0, 1], [0, 1], [0, 1], [0, 0], [0, 0], [0, 0], [0, 1], [0, 1]])]

    seq = func(seq, 3)
    assert are_lists_of_tensors_equal(seq, res)


    seq = torch.tensor([[0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 1]])
    res = [torch.tensor([[0, 1], [0, 0], [0, 0]]), torch.tensor([[0, 1], [0, 1], [0, 0], [0, 0]]),
           torch.tensor([[0, 1]])]

    seq = func(seq, 2)
    assert are_lists_of_tensors_equal(seq, res)

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

    seq = torch.tensor(
        [[0, 1], [0, 0], [0, 1], [0, 1], [0, 0], [0, 0], [0, 0], [0, 1], [0, 0], [0, 0], [0, 1], [0, 0], [0, 0], [0, 0],
         [0, 0], [0, 0]])
    res = [torch.tensor([[0, 1], [0, 0], [0, 1], [0, 1], [0, 0], [0, 0]]),
           torch.tensor([[0, 1], [0, 0], [0, 0], [0, 1], [0, 0], [0, 0]])]

    seq = func(seq, 2)
    assert are_lists_of_tensors_equal(seq, res)

    seq = torch.tensor(
        [[0, 1], [0, 0], [0, 1], [0, 1], [0, 0], [0, 0], [0, 0], [0, 1], [0, 0], [0, 0], [0, 1], [0, 0], [0, 0], [0, 0],
         [0, 0], [0, 0]])
    res = [torch.tensor(
        [[0, 1], [0, 0], [0, 1], [0, 1], [0, 0], [0, 0], [0, 0], [0, 1], [0, 0], [0, 0], [0, 1], [0, 0], [0, 0],
         [0, 0]])]

    seq = func(seq, 3)
    assert are_lists_of_tensors_equal(seq, res)

    print("Success")
    sys.exit(0)


if __name__ == "__main__":
    test_split_tensor(split_tensor_gpu)

    seq = [[(0, i) for i in range(3)], [(2, i) for i in range(2)], [(4, i) for i in range(3)],
           [(5, i) for i in range(2)], [(6, i) for i in range(1)], [(7, i) for i in range(4)],
           [(8, i) for i in range(3)], [(9, i) for i in range(3)], [(10, i) for i in range(1)]]
    test = split_by(seq, [0.7, 0.2, 0.1])
    print(test)
