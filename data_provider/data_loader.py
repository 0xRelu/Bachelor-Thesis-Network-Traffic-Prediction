import ast
import datetime
import itertools
import math
import pickle
import random
import sys

import numpy as np
import pandas as pd
import os

import scipy.signal
import torch
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d, convolve
from scipy.signal import stft
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from utils.data_perperation import split_by
from utils.scaler import StandardScalerNp, StandardScalerList
from utils.timefeatures import time_features
import warnings

from utils.tools import parse_unix_time, ema_smoothing

warnings.filterwarnings('ignore')


class Dataset_Test(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', stride=1000, transform=False, smooth_param=None):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

    def __getitem__(self, index):
        seq_x = torch.zeros((self.seq_len, 1))
        seq_y = torch.zeros((self.label_len + self.pred_len, 1))

        seq_x_mark = torch.zeros((self.seq_len, 1))
        seq_y_mark = torch.zeros((self.seq_len, 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return 500


class Dataset_Traffic_Singe_Packets(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='univ1_pt1_new_csv.csv',
                 target='OT', scale=True, timeenc=0, freq='h', stride=100, transform=False, smooth_param=None):
        # size [seq_len, label_len, pred_len]
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        self.flag = flag
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]

        self.root_path = root_path
        self.data_path = data_path[1:] if data_path.startswith('/') or data_path.startswith('\\') else data_path

        self.random_seed = 100
        self.scaler = None

        self.__read_data__()

    def __read_data__(self):
        save_path = os.path.join(self.root_path, self.data_path)
        with open(save_path, 'rb') as f:
            data = pickle.load(f)

        # prepare data if seq_len, label_len or pred_len does not match
        if 'shape' not in data or len(data['shape']) != 3 or \
                data['shape'][0] != self.seq_len or data['shape'][1] != self.label_len \
                or data['shape'][2] != self.pred_len:
            print(f"[-] Shape of saved data does not match "
                  f"new configuration ({self.seq_len}, {self.label_len}, {self.pred_len}). Will try do prepare data...")
            raise FileExistsError("Expected File did not exist")
            # data = self.__prepare_data__()
        else:
            print(f"[+] Shape of saved data {data['shape']} matches current "
                  f"configuration ({self.seq_len}, {self.label_len}, {self.pred_len}). Will use saved data...")

        flow_seq_x = data['data_x']
        flow_seq_y = data['data_y']

        # split in test and train
        seq_count = sum(map(lambda x: len(x[0]), flow_seq_x))  # len(x) == len(y)
        flow_seq_x = sorted(flow_seq_x, key=lambda x: len(x[0]), reverse=True)
        flow_seq_y = sorted(flow_seq_y, key=lambda x: len(x[0]), reverse=True)

        train_size = 0.8 * seq_count
        train_seq_x = []
        train_seq_y = []

        test_seq_x = []
        test_seq_y = []

        for i in range(len(flow_seq_x)):
            if sum(map(lambda x: len(x[0]), train_seq_x)) > train_size:
                test_seq_x = flow_seq_x[i:]
                test_seq_y = flow_seq_y[i:]
                break

            train_seq_x.append(flow_seq_x[i])
            train_seq_y.append(flow_seq_y[i])

        # normalize
        total_length = sum(map(lambda x: x[-1], train_seq_x))  # != seq_count
        mean = sum([(f[-1] / total_length) * f[1] for f in train_seq_x])
        std = math.sqrt(sum([(f[-1] / total_length) * f[2] for f in train_seq_x]))

        train_seq_x = np.concatenate(list(map(lambda x: x[0], train_seq_x)))
        train_seq_y = np.concatenate(list(map(lambda x: x[0], train_seq_y)))

        test_seq_x = np.concatenate(list(map(lambda x: x[0], test_seq_x)))
        test_seq_y = np.concatenate(list(map(lambda x: x[0], test_seq_y)))

        self.scaler = StandardScalerNp(mean=mean, std=std)
        train_seq_y[:, :, 1] = self.scaler.transform(train_seq_y[:, :, 1])
        test_seq_y[:, :, 1] = self.scaler.transform(test_seq_y[:, :, 1])

        # scaler = MinMaxScalerNp()
        # train_seq_y[:, :, 1] = scaler.fit_transform(train_seq_y[:, :, 1])
        # test_seq_y[:, :, 1] = scaler.transform(test_seq_y[:, :, 1])

        val_x, test_x = test_seq_x, test_seq_x
        val_y, test_y = test_seq_y, test_seq_y

        self.seq_x = [train_seq_x, val_x, test_x][self.set_type]
        self.seq_y = [train_seq_y, val_y, test_y][self.set_type]

    def __getitem__(self, index):
        # <<<< x >>>>
        seq_x = self.seq_x[index, :, 1:]  # dont add time feature
        seq_x_mark = parse_unix_time(self.seq_x[index, :, 0])

        # <<<< y >>>>
        seq_y = self.seq_y[index, :, 1:]
        seq_y_mark = parse_unix_time(self.seq_y[index, :, 0])

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.seq_x)  # len(self.seq_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Traffic_Even(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='univ1_pt1_new_csv.csv',
                 target='OT', scale=True, timeenc=0, freq='h', stride=100, transform=False, smooth_param=None):
        # size [seq_len, label_len, pred_len]
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]

        self.root_path = root_path
        self.data_path = data_path

        self.random_seed = 100
        self.scaler = None

        self.__read_data__()

    def __read_data__(self):
        save_path = os.path.join(self.root_path, self.data_path)
        with open(save_path, 'rb') as f:
            data = pickle.load(f)

        # prepare data if seq_len, label_len or pred_len does not match
        if 'shape' not in data or len(data['shape']) != 3 or \
                data['shape'][0] != self.seq_len or data['shape'][1] != self.label_len \
                or data['shape'][2] != self.pred_len:
            print(f"[-] Shape of saved data does not match "
                  f"new configuration ({self.seq_len}, {self.label_len}, {self.pred_len}). Will try do prepare data...")
            raise FileExistsError("Expected File did not exist")
            # data = self.__prepare_data__()
        else:
            print(f"[+] Shape of saved data {data['shape']} matches current "
                  f"configuration ({self.seq_len}, {self.label_len}, {self.pred_len}). Will use saved data...")

        # train, val_test = data['train'], data['test']

        flow_seq = data['data']

        # split in test and train
        seq_count = sum(map(lambda x: len(x[0]), flow_seq))

        random.seed(self.random_seed)
        random.shuffle(flow_seq)
        # flow_seq = sorted(flow_seq, key=lambda x: len(x[0]), reverse=True)

        train_size = 0.8 * seq_count
        train_seq = []
        test_seq = []

        for i in range(len(flow_seq)):
            if sum(map(lambda x: len(x[0]), train_seq)) > train_size:
                test_seq = flow_seq[i:]
                break

            train_seq.append(flow_seq[i])

        # normalize
        total_length = sum(map(lambda x: x[-1], train_seq))  # != seq_count
        mean = sum([(f[-1] / total_length) * f[1] for f in train_seq])
        std = math.sqrt(sum([(f[-1] / total_length) * f[2] for f in train_seq]))

        train_seq = np.concatenate(list(map(lambda x: x[0], train_seq)))
        test_seq = np.concatenate(list(map(lambda x: x[0], test_seq)))

        # we don't need to fit because we calc mean from flow to make it independent of pred_len
        self.scaler = StandardScalerNp(mean=mean, std=std)
        train_seq[:, :, 1] = self.scaler.transform(train_seq[:, :, 1])
        test_seq[:, :, 1] = self.scaler.transform(test_seq[:, :, 1])

        train, val, test = train_seq, test_seq, test_seq
        self.seq = [train, val, test][self.set_type]
        self.time_stamp = np.stack([parse_unix_time(self.seq[i, :, 0]) for i in range(len(self.seq))])

    def __getitem__(self, index):
        # <<<< x >>>>
        seq_x = self.seq[index, :self.seq_len, 1:]
        seq_x_mark = self.time_stamp[index, :self.seq_len]

        # <<<< y >>>>
        seq_y = self.seq[index, -self.pred_len - self.label_len:, 1:]
        seq_y_mark = self.time_stamp[index, -self.pred_len - self.label_len:]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.seq)

    def inverse_transform(self, data):
        rdata = self.scaler.inverse_transform(data)
        # reverse FFT
        # for i in range(rdata.shape[0]):
        #    rdata[i, :, 0] = np.fft.fft(rdata[i, :, 0])  # no time feature

        return rdata


class Dataset_Traffic_Even_n(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='univ1_pt1_even.csv',
                 target='OT', scale=True, timeenc=0, freq='h', stride=100, transform=None, smooth_param=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.transform = transform
        self.smooth_param = smooth_param  # gaussian, uniform and ema
        self.stride = stride

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        self.border1 = border1s[self.set_type]
        self.border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        data = df_data.values  # date, bytes 

        if self.transform == 'stft':
            '''
            1. Get params and copy data
            2. Transform time series with no boundaries (-> not paading)
            3. Get borders for new time series x and for copied time series y. Get them from x and "convert" it into index in time domain
            4. Normalize y
            5. Normalize x
            6. Parse time stamps
            7. Get Data_x, data_y, date_stamp_x and date_stamp_y
            '''
            seg_len, seg_overlap = tuple(ast.literal_eval(self.smooth_param))
            data_y = data.copy()

            # stft
            self.stft_f, self.stft_time, Zxx = stft(data.flatten(), nperseg=seg_len, noverlap=seg_overlap,
                                                    boundary=None)
            # we get [x, 51] remember we have a segment length of 100! So we did expect [x, 100]
            # so we split in real an imag -> and get [x, 102]

            data = Zxx.transpose()
            data = np.concatenate((data.real, data.imag), axis=1)

            num_train = int(len(data) * 0.7)
            num_test = int(len(data) * 0.2)
            num_vali = len(data) - num_train - num_test
            border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(data)]
            self.border1 = border1s[self.set_type]
            self.border2 = border2s[self.set_type]
            self.border1_y = int(self.stft_time[self.border1])
            self.border2_y = int(self.stft_time[self.border2]) if self.border2 != len(data) else len(data_y)

            train_data_y = data_y[int(self.stft_time[border1s[0]]): int(self.stft_time[border2s[0]])]
            self.scaler.fit(train_data_y)
            data_y = self.scaler.transform(data_y)

        if self.transform == 'gaussian':
            data = gaussian_filter1d(data.reshape(-1), sigma=self.smooth_param, mode="nearest").reshape(-1, 1)

        if self.transform == 'uniform':
            kernel = np.array([1 / self.smooth_param for _ in range(-((self.smooth_param - 1) // 2),
                                                                    ((self.smooth_param - 1) // 2))])
            data = convolve(data.reshape(-1), weights=kernel, mode="constant", cval=0.0).reshape(-1, 1)

        if self.transform == 'ema':
            data = ema_smoothing(data.reshape(-1), a=self.smooth_param).reshape(-1, 1)

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        df_stamp = df_raw[['date']][self.border1:self.border2] if self.transform != 'stft' else df_raw[['date']][
                                                                                                self.border1_y: self.border2_y]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['second'] = df_stamp.date.apply(lambda row: row.second, 1)
            df_stamp['millisecond'] = df_stamp.date.apply(lambda row: row.microsecond // 1000, 1)
            df_stamp['microsecond'] = df_stamp.date.apply(lambda row: row.microsecond % 1000, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        if self.transform != 'stft':
            self.data_x = data[self.border1:self.border2]
            self.data_y = data[self.border1:self.border2]
            self.data_stamp_x = data_stamp
            self.data_stamp_y = data_stamp
            self.index = np.arange(0, len(self.data_x) - self.seq_len - self.pred_len + 1, self.stride)
        else:
            self.data_x = data[self.border1:self.border2]
            self.data_y = data_y[self.border1_y:self.border2_y]

            self.stft_time = self.stft_time.astype(int)[self.border1: self.border2]
            self.stft_time = self.stft_time - self.stft_time[0]

            self.data_stamp_x = data_stamp[self.stft_time]  # delete last
            self.data_stamp_y = data_stamp
            self.index = np.arange(0, len(self.data_x) - self.seq_len - self.pred_len + 1, self.stride)

        if len(self.data_x) != len(self.data_stamp_x) or len(self.data_y) != len(self.data_stamp_y):
            raise IndexError("Has to be of the same size")

    def __getitem__(self, index):
        index = self.index[index]

        if self.transform == 'stft':
            s_begin = index
            s_end = s_begin + self.seq_len
            r_begin = int(self.stft_time[s_end] - self.label_len)
            r_end = r_begin + self.label_len + self.pred_len
        else:
            s_begin = index
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp_x[s_begin:s_end]
        seq_y_mark = self.data_stamp_y[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.index)  # len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Traffic_Even_n2(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='univ1_pt1_even.csv',
                 target='OT', scale=True, timeenc=0, freq='h', stride=100, transform=None, smooth_param=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.transform = transform
        self.smooth_param = smooth_param  # gaussian, uniform and ema
        self.stride = stride

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScalerList()

        with open(os.path.join(self.root_path, self.data_path), 'rb') as f:
            data_raw: list[list] = pickle.load(f)  # returns list[list]

        print(f"[+] Loaded {len(data_raw)} flows.")

        indexes = []  # sequences

        data_stamps = []
        data = []

        for i in range(len(data_raw)):
            if len(data_raw[i]) < self.seq_len + self.pred_len:
                continue

            data_stamp = np.array(list(map(lambda x: x[0], data_raw[i])))
            data_bytes_flow = np.array(list(map(lambda x: x[1], data_raw[i]))).reshape(-1, 1)

            if self.transform == 'gaussian':
                data_bytes_flow = gaussian_filter1d(data_bytes_flow.reshape(-1), sigma=self.smooth_param,
                                                    mode="nearest").reshape(-1, 1)

            if self.transform == 'uniform':
                kernel = np.array([1 / self.smooth_param for _ in range(-((self.smooth_param - 1) // 2),
                                                                        ((self.smooth_param - 1) // 2))])
                data_bytes_flow = convolve(data_bytes_flow.reshape(-1), weights=kernel, mode="constant",
                                           cval=0.0).reshape(-1, 1)

            if self.transform == 'ema':
                data_bytes_flow = ema_smoothing(data_bytes_flow.reshape(-1), a=self.smooth_param).reshape(-1, 1)

            indexes.append([[len(data), j] for j in range(len(data_bytes_flow) - self.seq_len - self.pred_len)])

            data.append(data_bytes_flow)
            data_stamps.append(data_stamp)

        print(f"[+] Found {sum([len(x) for x in data])} sequences in {len(data_raw)} flows.")

        assert len(data_stamps) == len(data)

        splitted_index = split_by(indexes, [0.7, 0.1, 0.2])
        assert len(splitted_index) == 3
        border1s = [splitted_index[i][0][0][0] for i in range(len(splitted_index))]
        border2s = [splitted_index[i][-1][0][0] + 1 for i in
                    range(len(splitted_index))]  # +1 because we also want the content of the key
        self.border1 = border1s[self.set_type]
        self.border2 = border2s[self.set_type]

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        self.data_x = data[self.border1: self.border2]
        self.data_y = data[self.border1: self.border2]
        self.data_stamp_x = data_stamps[self.border1: self.border2]
        self.data_stamp_y = data_stamps[self.border1: self.border2]
        indexes = [(a[0] - indexes[self.border1][0][0], a[1]) for a in
                   itertools.chain(*indexes[self.border1: self.border2])]
        self.index = [x for i, x in enumerate(indexes) if i % self.stride == 0]

    def __getitem__(self, index):
        index = self.index[index]

        s_begin = index[1]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[index[0]][s_begin:s_end]
        seq_y = self.data_y[index[0]][r_begin:r_end]
        seq_x_mark = self.data_stamp_x[index[0]][s_begin:s_end]
        seq_y_mark = self.data_stamp_y[index[0]][r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.index)  # len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Traffic_Even_nstft(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='univ1_pt1_even.csv',
                 target='OT', scale=True, timeenc=0, freq='h', stride=100, transform=None, smooth_param=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.transform = transform
        self.smooth_param = smooth_param  # gaussian, uniform and ema
        self.stride = stride

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScalerList()
        self.scaler_y = StandardScalerList()  # for stft

        with open(os.path.join(self.root_path, self.data_path), 'rb') as f:
            data_raw: list[list] = pickle.load(f)  # returns list[list]

        print(f"[+] Loaded {len(data_raw)} flows.")

        f = []

        indexes = []  # sequences
        indexes_y = []

        data_stamps = []
        data = []

        data_stamps_y = []
        data_y = []  # only needed for stft

        for i in range(len(data_raw)):
            if len(data_raw[i]) < self.seq_len + self.pred_len:
                continue

            data_stamp = np.array(list(map(lambda x: x[0], data_raw[i])))
            data_bytes_flow = np.array(list(map(lambda x: x[1], data_raw[i]))).reshape(-1, 1)

            data_stamp_y = data_stamp.copy()
            data_bytes_y = data_bytes_flow.copy()

            self.seg_len, self.seg_overlap = tuple(ast.literal_eval(self.smooth_param))
            # stft
            stft_f, stft_time, data_bytes_flow = stft(data_bytes_flow.flatten(), nperseg=self.seg_len,
                                                      noverlap=self.seg_overlap, boundary=None)
            data_bytes_flow = data_bytes_flow.transpose()

            if len(data_bytes_flow) < self.seq_len + (self.pred_len / (self.seg_len - self.seg_overlap)):
                continue

            data_bytes_flow = np.concatenate((data_bytes_flow.real, data_bytes_flow.imag), axis=1)
            data_stamp = data_stamp[stft_time.astype(int).tolist()]

            indexes.append([[len(data), j] for j in range(len(data_bytes_flow) - self.seq_len -
                                                          (self.pred_len // (self.seg_len - self.seg_overlap)))])
            indexes_y.append(stft_time.astype(int).tolist())

            f.append(stft_f)

            data.append(data_bytes_flow)
            data_stamps.append(data_stamp)

            data_stamps_y.append(data_stamp_y)
            data_y.append(data_bytes_y)

        print(f"[+] Found {sum([len(x) for x in data])} sequences in {len(data_raw)} flows.")
        # data_stamps = np.array(data_stamps)
        # data = np.stack(data)

        assert len(data_stamps) == len(data)
        assert len(data) == len(data_y)

        splitted_index = split_by(indexes, [0.7, 0.1, 0.2])
        assert len(splitted_index) == 3
        border1s = [splitted_index[i][0][0][0] for i in range(len(splitted_index))]
        border2s = [splitted_index[i][-1][0][0] + 1 for i in
                    range(len(splitted_index))]  # +1 because we also want the content of the key
        self.border1 = border1s[self.set_type]
        self.border2 = border2s[self.set_type]

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

            train_data_y = data_y[border1s[0]:border2s[0]]  # gets the same flows as data[...]
            self.scaler_y.fit(train_data_y)
            data_y = self.scaler_y.transform(data_y)

        self.data_x = data[self.border1: self.border2]
        self.data_y = data_y[self.border1: self.border2]
        self.data_stamp_x = data_stamps[self.border1: self.border2]
        self.data_stamp_y = data_stamps_y[self.border1: self.border2]

        indexes = [(a[0] - indexes[self.border1][0][0], a[1]) for a in
                   itertools.chain(*indexes[self.border1: self.border2])]
        self.index = [x for i, x in enumerate(indexes) if i % self.stride == 0]
        self.index_y = indexes_y[self.border1: self.border2]

        self.f = f[self.border1: self.border2]

    def __getitem__(self, index):
        index = self.index[index]

        s_begin = index[1]
        s_end = s_begin + self.seq_len
        r_begin = self.index_y[index[0]][s_end] - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[index[0]][s_begin:s_end]
        seq_y = self.data_y[index[0]][r_begin:r_end]
        seq_x_mark = self.data_stamp_x[index[0]][s_begin:s_end]
        seq_y_mark = self.data_stamp_y[index[0]][r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.index)  # len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):  # can only transform target not context/input!!!
        return self.scaler_y.inverse_transform(data)


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', stride=1000, transform=False, smooth_param=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        self.random_index = [random.randint(border1, border2 - self.seq_len - self.pred_len + 1) for _ in range(10)]

    def __getitem__(self, index):
        index = self.random_index[index]

        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return 10  # len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', stride=1000, transform=False, smooth_param=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', stride=1000, transform=False, smooth_param=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None, stride=1000,
                 transform=False, smooth_param=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
