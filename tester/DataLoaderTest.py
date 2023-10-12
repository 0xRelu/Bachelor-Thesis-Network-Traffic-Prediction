import sys

import numpy as np
import torch

from data_provider.data_factory import data_provider
from utils.tools import dotdict

if __name__ == "__main__":
    print("Start")

    cw_config = {
        'data': 'Traffic_Single',
        'batch_size': 128,
        'freq': "h",
        'root_path': "C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1",
        'data_path': "univ1_pt1_single_2_336_48_30_1000.pkl",
        'seq_len': 336,
        'label_len': 48,
        'pred_len': 30,
        'features': "M",
        'target': "M",
        'num_workers': 1,
        'embed': 'timeF',
        'random_seed': 12,
        'use_minmax_scaler': False
    }

    config = dotdict(cw_config)

    train_data, train_loader = data_provider(config, flag='train')  # , collate_fn=padded_collate_fn)
    print("Length: ", len(train_data))

    test_data, test_loader = data_provider(config, flag='test')
    print("Length: ", len(test_data))

    val_data, val_loader = data_provider(config, flag='val')
    print("Length: ", len(val_data))

    max_x = 0
    max_y = 0

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        if i % 100 == 0:
            print(f"\tTrain {i / len(train_loader)} ")

        nmax = torch.max(batch_x)
        if nmax > max_x:
            max_x = nmax

        nmax = torch.max(batch_y)
        if nmax > max_y:
            max_y = nmax

    print(f"{max_x} | {max_y}")

    max_x = 0
    max_y = 0

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        if i % 100 == 0:
            print(f"\tTest {i / len(test_loader)}")

        nmax = torch.max(batch_x)
        if nmax > max_x:
            max_x = nmax

        nmax = torch.max(batch_y)
        if nmax > max_y:
            max_y = nmax

    print(f"{max_x} | {max_y}")

    max_x = 0
    max_y = 0

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(val_loader):
        if i % 100 == 0:
            print(f"\tVal {i / len(val_loader)}")

        nmax = torch.max(batch_x)
        if nmax > max_x:
            max_x = nmax

        nmax = torch.max(batch_y)
        if nmax > max_y:
            max_y = nmax

    print(f"{max_x} | {max_y}")
