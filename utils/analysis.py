import datetime
import os
import pickle
import random
import sys

import numpy as np
import scipy.stats
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
from torch import tensor
from statsmodels.sandbox.stats.runs import runstest_1samp
import statsmodels.api as sm
from utils.data_preparation_tools import split_tensor_gpu


def visualize_test():
    p = np.array([25, 25, 50, 0, 25, 0, 0, 0, 0, 0, 0,
                  50, 25, 25, 0, 75, 0, 25, 0, 75, 0, 0, 0, 0, 0,
                  50, 0, 50, 0, 50, 0, 50, 25, 25, 0, 0, 0, 0, 0, 0,
                  50, 25, 25, 0, 75, 0, 25, 0, 75, 0, 0, 0, 0, 0,
                  50, 0, 25, 25, 50, 0, 75, 0, 0, 0, 0, 0, 0, 0, 0,
                  50, 0, 50, 0, 75, 0, 25, 0, 75, 0, 0, 0, 0, 0, 0, 0,
                  50, 0, 50, 0, 75, 0, 25, 0, 75, 0, 0, 0, 0, 0, 0, ])
    t = np.linspace(0, 1, 100)

    print(t.tolist())
    print(p.tolist())

    plt.plot(t, p)
    plt.xlabel('time (ms)')
    plt.show()

    p = np.abs(np.fft.fft(p))
    print("----")
    print(p.tolist())

    plt.plot(p)
    plt.xlabel('time (ms)')
    plt.show()

    t, f, p = scipy.signal.stft(p, nperseg=3, noverlap=2)
    p = p.transpose()

    print("----")

    for i in range(p.shape[1]):
        k = p[:, i].real
        print(k.tolist())
        plt.plot(k)
        plt.title(str(i) + ' REAL')
        plt.xlabel('time (ms)')
        plt.show()

        j = p[:, i].imag
        print(j.tolist())
        plt.plot(j)
        plt.title(str(i) + ' IMAG')
        plt.xlabel('time (ms)')
        plt.show()


def visualize_flows(file_path, aggr=1000, min_length=2500, skip=0, amount=20, filter_tcp=True, shuffle=12):
    with open(file_path, 'rb') as f:
        data_flows = pickle.load(f)

    keys = list(data_flows.keys())

    if filter_tcp:
        keys = [k for k in keys if k[0].startswith('TCP')]
        print(f"[+] Found {len(keys)} flows with filter tcp.")

    if shuffle is not None:
        random.seed(shuffle)
        random.shuffle(keys)

    keys = keys[skip:]

    counter = 0
    i = 0

    while counter < amount:
        k = keys[i]
        i += 1

        flow = torch.tensor([x[:2] for x in data_flows[k]], dtype=torch.float64)
        start_time = int(flow[0, 0] * aggr)  # assumes packets are ordered
        end_time = int(flow[-1, 0] * aggr) + 1
        flow_series_bytes = torch.zeros(end_time - start_time + 1, dtype=torch.float64)
        flow_series_time = torch.arange(start_time, end_time + 1, dtype=torch.float64)

        if len(flow_series_bytes) < min_length:
            continue

        counter += 1
        packet_times = ((flow[:, 0] * aggr) - start_time).long()
        flow_series_bytes.index_add_(0, packet_times, flow[:, 1])
        plt.plot(flow_series_bytes.tolist(), label=k)
        plt.legend()
        plt.show()

        # flow_series_bytes = torch.stack((flow_series_time / aggr, flow_series_bytes), dim=1)
        # flow_list = split_tensor_gpu(flow_series_bytes, consecutive_zeros=consecutive_zeros)
        #
        # flow_list = [f for f in flow_list if f.shape[0] > min_length]
        #
        # for f in flow_list:
        #     flow = [x[1] for x in f]
        #     plt.plot(flow, label=k)
        #     plt.legend()
        #     plt.show()


def visualize_flows_stft(file_path, aggr=1000, consecutive_zeros=500, min_length=1000, amount=-1, filter_tcp=True,
                         shuffle=12):
    with open(file_path, 'rb') as f:
        data_flows = pickle.load(f)

    keys = list(data_flows.keys())

    if filter_tcp:
        keys = [k for k in keys if k[0].startswith('TCP')]
        print(f"[+] Found {len(keys)} flows with filter tcp.")

    if shuffle is not None:
        random.seed(shuffle)
        random.shuffle(keys)

    keys = keys[:amount]

    for k in keys:
        flow = torch.tensor([x[:2] for x in data_flows[k]], dtype=torch.float64)
        start_time = int(flow[0, 0] * aggr)  # assumes packets are ordered
        end_time = int(flow[-1, 0] * aggr) + 1
        flow_series_bytes = torch.zeros(end_time - start_time + 1, dtype=torch.float64)
        flow_series_time = torch.arange(start_time, end_time + 1, dtype=torch.float64)

        if len(flow_series_bytes) < min_length:
            continue

        packet_times = ((flow[:, 0] * aggr) - start_time).long()
        flow_series_bytes.index_add_(0, packet_times, flow[:, 1])
        # flow_series_bytes = torch.stack((flow_series_time / aggr, flow_series_bytes), dim=1)

        non_zero = torch.count_nonzero(flow_series_bytes)

        if non_zero < len(flow_series_bytes) * 0.4:
            continue

        plt.plot(flow_series_bytes)
        plt.title('load (bytes)')
        plt.xlabel('time (ms)')
        plt.show()

        f = torch.stft(flow_series_bytes, n_fft=16, hop_length=2, center=False, pad_mode="reflect", normalized=True,
                       return_complex=True, onesided=True)

        f = f.permute(1, 0)

        for i in range(f.shape[1]):
            plt.plot(f[:, i])
            plt.title('')
            plt.xlabel('time (ms)')
            plt.show()

        # flow_list = split_tensor_gpu(flow_series_bytes, consecutive_zeros=consecutive_zeros)
        #
        # flow_list = [f for f in flow_list if f.shape[0] > min_length]
        #
        # for f in flow_list:
        #     f = f[:1000, 1]
        #
        #     # Plotten
        #     plt.plot(f)
        #     plt.title('')
        #     plt.xlabel('time (ms)')
        #     plt.ylabel('load (kbit)')
        #     plt.show()
        #
        #     f = np.abs(np.fft.fft(f.numpy()))
        #     fft_result_magnitude = np.abs(f)[:500]
        #     frequencies = np.fft.fftfreq(len(f), 1 / 1000)[:500]
        #
        #     plt.plot(frequencies, fft_result_magnitude)
        #     plt.title('')
        #     plt.xlabel('amplitude (unit)')
        #     plt.ylabel('frequency (Hz)')
        #     plt.show()


def visualize_acf(file_path, aggr=1000, amount=20, nlags=1000, filter_tcp=True, shuffle=True, min_length=None):
    with open(file_path, 'rb') as f:
        data_flows = pickle.load(f)

    keys = list(data_flows.keys())

    def calc_duration(x):
        start = datetime.datetime.fromtimestamp(x[0][0])
        end = datetime.datetime.fromtimestamp(x[-1][0])
        return (end - start).total_seconds()

    if filter_tcp:
        keys = [k for k in keys if k[0].startswith('TCP')]
        print(f"[+] Found {len(keys)} flows with filter tcp.")

    if shuffle:
        random.seed(122)
        random.shuffle(keys)

    if min_length is not None:
        nkeys = []
        i = 0
        while len(nkeys) < amount:
            dur = calc_duration(data_flows[keys[i]])
            if min_length < dur < 3 * min_length:
                nkeys.append(keys[i])
            i += 1

        keys = nkeys
        print(f"[+] Found {len(keys)} flows with min length: {min_length}.")

    keys = keys[:amount]

    for k in keys:
        flow = torch.tensor([x[:2] for x in data_flows[k]], dtype=torch.float64)
        start_time = int(flow[0, 0] * aggr)  # assumes packets are ordered
        end_time = int(flow[-1, 0] * aggr) + 1
        flow_series_bytes = torch.zeros(end_time - start_time + 1, dtype=torch.float64)
        packet_times = ((flow[:, 0] * aggr) - start_time).long()
        flow_series_bytes.index_add_(0, packet_times, flow[:, 1])

        flow_series_bytes = flow_series_bytes.tolist()

        plot_acf(flow_series_bytes, lags=range(1, min(1000, len(flow_series_bytes) - 1)), alpha=0.05, auto_ylims=True,
                 zero=False)
        acf_v = acf(flow_series_bytes, nlags=min(1000, len(flow_series_bytes) - 1))
        print(acf_v.tolist())
        plt.title("")
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation Coefficient')
        plt.show()


def count_packets(file_path):
    with open(file_path, 'rb') as f:
        data_flows = pickle.load(f)

    counter = 0
    for k, v in data_flows.items():
        counter += len(v)

    print(counter)


def filter_flows(file_path, save_path, alpha=0.05, aggr=1000, filter_tcp=True, auto=False):
    with open(file_path, 'rb') as f:
        data_flows = pickle.load(f)

    keys = list(data_flows.keys())

    if filter_tcp:
        keys = [k for k in keys if k[0].startswith('TCP')]
        print(f"[+] Found {len(keys)} flows with filter tcp.")

    counter = 0

    auto_mean = []

    for k in keys:
        if counter % 1000 == 0:
            print(f"[+] Processed {counter / len(keys)}.")

        counter += 1

        flow = torch.tensor([x[:2] for x in data_flows[k]], dtype=torch.float64)
        start_time = int(flow[0, 0] * aggr)  # assumes packets are ordered
        end_time = int(flow[-1, 0] * aggr) + 1
        flow_series_bytes = torch.zeros(end_time - start_time + 1, dtype=torch.float64)

        if len(flow_series_bytes) < 1000:
            continue

        packet_times = ((flow[:, 0] * aggr) - start_time).long()
        flow_series_bytes.index_add_(0, packet_times, flow[:, 1])

        flow_series_bytes = flow_series_bytes.numpy()
        p = sm.tsa.acf(flow_series_bytes, nlags=min(len(flow_series_bytes), 1000))
        p = np.abs(p).mean()
        auto_mean.append([k, p, len(flow_series_bytes)])

    auto_mean = sorted(auto_mean, key=lambda x: abs(x[1]), reverse=True)
    auto_mean = [x for x in auto_mean if abs(x[1]) >= alpha]

    data_flows = {x[0]: data_flows[x[0]] for x in auto_mean}

    # save results
    with open(save_path, 'wb') as f:
        pickle.dump(data_flows, f)


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
    p = data_flows[data_flows[:, 7] >= 0][:, 7].mean()
    p_none = len(data_flows[data_flows[:, 7] < 0])

    p_amount = data_flows[(data_flows[:, 7] <= 0.05) & (data_flows[:, 7] >= 0)]

    l01 = data_flows[data_flows[:, 2] < 0.1]
    l011 = data_flows[(0.1 <= data_flows[:, 2]) & (data_flows[:, 2] < 1)]
    l110 = data_flows[(1 <= data_flows[:, 2]) & (data_flows[:, 2] < 10)]
    l10100 = data_flows[(10 <= data_flows[:, 2]) & (data_flows[:, 2] < 100)]
    l100 = data_flows[data_flows[:, 2] >= 100]

    print(f"ptm: {ptm}, vtm: {vtm}, b: {b}, a:{a}, p:{p}|{len(p_amount) / len(data_flows)}|{p_none / len(data_flows)}")
    print(f"l01: {len(l01)}|{sum([x[1] for x in l01])}, l011: {len(l011)}|{sum([x[1] for x in l011])}, "
          f"l110: {len(l110)}|{sum([x[1] for x in l110])}, l10100: {len(l10100)}|{sum([x[1] for x in l10100])}, "
          f"l10100: {len(l100)}|{sum([x[1] for x in l100])}")
    print(f"{len(l110) + len(l10100) + len(l100)}")


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
        _, p = runstest_1samp(flow_series_bytes.tolist(), correction=False)

        if np.isnan(p):
            p = -1

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

            a.append((torch.sqrt(torch.tensor(sub_flow.shape[0], device=device) + 1) * vtm_ - torch.sqrt(
                torch.tensor(sub_flow.shape[0], device=device) - 1)) /
                     ((torch.sqrt(torch.tensor(sub_flow.shape[0], device=device) + 1) - 2) * vtm_ + torch.sqrt(
                         torch.tensor(sub_flow.shape[0], device=device) - 1)))
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
                      torch.tensor(b).mean().item(), torch.tensor(a).mean().item(), p, 0, 0]
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

    filter_path = 'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1_n\\univ1_pt_filtered.pkl'
    filter_path_100 = 'C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1_n\\univ1_pt_filtered_100.pkl'

    # count_packets(path)

    # visualize_flows(test_path, aggr=1000, skip=30, amount=20, min_length=500)
    # visualize_acf(filter_path, aggr=1000)
    # analysis_gpu(path, save_analysis, aggr=1000)
    # visualize_flows_stft(filter_path, aggr=10, min_length=10)
    visualize_test()
    # filter_flows(path, filter_path_100, aggr=100, alpha=0.02)

    print("<<<<<<<<<<<<<<<< Done >>>>>>>>>>>>>>>>")
