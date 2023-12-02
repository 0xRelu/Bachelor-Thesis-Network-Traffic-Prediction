import datetime
import os
import pickle

import torch
from matplotlib import pyplot as plt
from torch import tensor

from utils.data_preparation_tools import split_tensor_gpu


def visualize_flows(file_path, aggr=1000, amount=20, filter_tcp=True, get_max_Load=0.0005):
    with open(file_path, 'rb') as f:
        data_flows = pickle.load(f)

    keys = list(data_flows.keys())

    if filter_tcp:
        keys = [k for k in keys if k[0].startswith('TCP')]
        print(f"[+] Found {len(keys)} flows with filter tcp.")

    if get_max_Load is not None:
        keys_load = [(k, sum([x[1] for x in data_flows[k]])) for k in keys]
        keys_load = sorted(keys_load, key=lambda x: x[1], reverse=True)
        max_load = keys_load[0][1] * get_max_Load
        keys = [k[0] for k in keys_load if k[1] > max_load]

    keys = keys[:amount]

    for k in keys:
        flow = torch.tensor([x[:2] for x in data_flows[k]], dtype=torch.float64)
        start_time = int(flow[0, 0] * aggr)  # assumes packets are ordered
        end_time = int(flow[-1, 0] * aggr) + 1
        flow_series_bytes = torch.zeros(end_time - start_time + 1, dtype=torch.float64)
        flow_series_time = torch.arange(start_time, end_time + 1, dtype=torch.float64)

        packet_times = ((flow[:, 0] * aggr) - start_time).long()
        flow_series_bytes.index_add_(0, packet_times, flow[:, 1])
        flow_series_bytes = torch.stack((flow_series_time / aggr, flow_series_bytes), dim=1)
        flow_list = split_tensor_gpu(flow_series_bytes, consecutive_zeros=500)

        flow_list = [f for f in flow_list if f.shape[0] > 2 * 500]

        for f in flow_list:
            flow = [x[1] for x in f]
            plt.plot(flow, label=k)
            plt.legend()
            plt.show()


def _compare(new_file_path, old_file_path):
    with open(new_file_path, 'rb') as f:
        new_data = pickle.load(f)

    with open(old_file_path, 'rb') as f:
        old_data = pickle.load(f)

    nnew_data = []

    for k, v in new_data.items():
        if k[0].startswith('TCP'):
            nnew_data.append(v)

    assert len(nnew_data) == len(old_data)

    new_data = sorted(nnew_data, key=lambda x: len(x))
    old_data = sorted(old_data, key=lambda x: len(x))

    for i in range(len(new_data)):
        assert len(new_data[i]) == len(old_data[i])


def getData(data_flows: tensor, only_tcp=True):
    if only_tcp:
        data_flows = data_flows[data_flows[:, -1] == 0]  # 0 == TCP, 1 == UDP

    ptm = data_flows[:, 3].mean()
    vtm = data_flows[:, 4].mean()
    b = data_flows[:, 5].mean()
    a = data_flows[:, 6].mean()

    l01 = data_flows[data_flows[:, 2] < 0.1]
    l011 = data_flows[(0.1 <= data_flows[:, 2]) & (data_flows[:, 2] < 1)]
    l110 = data_flows[(1 <= data_flows[:, 2]) & (data_flows[:, 2] < 10)]
    l10100 = data_flows[(10 <= data_flows[:, 2]) & (data_flows[:, 2] < 100)]
    l100 = data_flows[data_flows[:, 2] >= 100]

    print(f"ptm: {ptm}, vtm: {vtm}, b: {b}, a:{a}")
    print(f"l01: {len(l01)}|{sum([x[1] for x in l01])}, l011: {len(l011)}|{sum([x[1] for x in l011])}, "
          f"l110: {len(l110)}|{sum([x[1] for x in l110])}, l10100: {len(l10100)}|{sum([x[1] for x in l10100])}, "
          f"l10100: {len(l100)}|{sum([x[1] for x in l100])}")


def analysis_gpu(file_path: str, save_path: str, aggr=1000, consecutive_zeros=500, only_tcp=True):
    open_path = file_path if not os.path.exists(save_path) else save_path

    with open(open_path, 'rb') as f:
        data_flows = pickle.load(f)

    if os.path.exists(save_path):
        getData(data_flows, only_tcp)
        return

    print("[+] Starting data analysis...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_flows_keys = list(data_flows.keys())
    data_flows = list(data_flows.values())

    for i in range(len(data_flows)):
        data_flows[i] = torch.tensor([x[:3] for x in data_flows[i]], device=device, dtype=torch.float64)

    flow_data = []

    for flow in data_flows:
        byte_series = flow[:, 1]

        packet_count = flow.shape[0]
        byte_count = byte_series.sum()

        start_time = int(flow[0, 0] * aggr)  # assumes packets are ordered
        end_time = int(flow[-1, 0] * aggr) + 1
        flow_series_bytes = torch.zeros(end_time - start_time + 1, device=device, dtype=torch.float64)

        packet_times = ((flow[:, 0] * aggr) - start_time).long()
        flow_series_bytes.index_add_(0, packet_times, flow[:, 1])
        flow_series_split = torch.stack((torch.zeros(flow_series_bytes.shape[0], device=device), flow_series_bytes),
                                        dim=1)
        flow_series_split = split_tensor_gpu(flow_series_split, consecutive_zeros)

        ptm = []
        vtm = []
        b = []
        a = []

        for sub_flow in flow_series_split:
            max_value = sub_flow[:, 1].max()
            mean_ = sub_flow[:, 1].mean()
            var_ = sub_flow[:, 1].var()

            vtm_ = var_ / mean_

            ptm.append(max_value / mean_)
            vtm.append(vtm_)
            b.append((var_ - mean_) / (mean_ + var_))

            a.append((torch.sqrt(torch.tensor(sub_flow.shape[0], device=device) + 1) * vtm_ - torch.sqrt(torch.tensor(sub_flow.shape[0], device=device) - 1)) /
                     ((torch.sqrt(torch.tensor(sub_flow.shape[0], device=device) + 1) - 2) * vtm_ + torch.sqrt(torch.tensor(sub_flow.shape[0], device=device) - 1)))
            # H, _, _ = calculate_hurst_exponent(sub_flow, device)

        if len(flow) < 2:
            duration = 0
        else:
            start_time = datetime.datetime.fromtimestamp(flow[0, 0].item())
            end_time = datetime.datetime.fromtimestamp(flow[-1, 0].item())
            duration = end_time - start_time

            duration = duration.total_seconds()

        nFlow_data = [packet_count, byte_count.item(), duration, torch.tensor(ptm).mean().item(),
                      torch.tensor(vtm).mean().item(),
                      torch.tensor(b).mean().item(), torch.tensor(a).mean().item(), 0, 0]
        flow_data.append(nFlow_data)

        if len(flow_data) % 1000 == 0:
            print(f"[+] Finished {len(flow_data) / len(data_flows)}")

    assert len(flow_data) == len(data_flows_keys)

    for i in range(len(flow_data)):
        flow_data[i][-1] = 0 if data_flows_keys[i][0].startswith('TCP') else 1

    flow_data = torch.tensor(flow_data)

    # save results
    with open(save_path, 'wb') as f:
        pickle.dump(flow_data, f)


if __name__ == "__main__":
    print("<<<<<<<<<<<<<<<< Start >>>>>>>>>>>>>>>>")
    test_path = 'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1_n\\univ1_pt_n_test.pkl'
    save_test_analysis = 'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\ANALYSIS\\analysis_test.pkl'

    path = 'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1_n\\univ1_pt_full.pkl'  # univ1_pt_n
    save_analysis = 'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\ANALYSIS\\analysis_full.pkl'  # _test

    visualize_flows(path)
    # analyse_flows(path, save_analysis)

    # analysis_gpu(path, save_analysis, 1000)

    print("<<<<<<<<<<<<<<<< Done >>>>>>>>>>>>>>>>")
