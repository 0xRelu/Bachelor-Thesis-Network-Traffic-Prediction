import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_provider.data_factory import data_provider
from utils.metrics import MSE
from utils.tools import dotdict

def plot_batches(batch_x: np.ndarray):
    x = list(range(batch_x.shape[1]))  # assume 3 dims

    for y in range(batch_x.shape[0]):
        plt.plot(batch_x[y], label=y)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    print("Start")

    cw_config = {
        'data': 'Traffic_Even_N',
        'batch_size': 128,
        'freq': "h",
        'root_path': "C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1_test",
        'data_path': "univ1_pt1_even_1000.pkl",  # univ1_pt1_even_336_48_12_1000.pkl
        'seq_len': 336,
        'label_len': 48,
        'pred_len': 96,
        'features': "M",
        'target': "bytes",
        'num_workers': 1,
        'embed': 'timeF',
        'random_seed': 12,
        'fourier_transform': False,
        'smooth': 'gaussian'
    }

    config = dotdict(cw_config)

    train_data, train_loader = data_provider(config, flag='train')  # , collate_fn=padded_collate_fn)
    print("Length: ", len(train_data))

    zero_counter_x = []
    zero_counter_y = []

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        if i % 100 == 0:
            print(f"\tTrain {i / len(train_loader)}")

        batch_x = np.stack([train_data.inverse_transform(x) for x in batch_x])
        batch_y = np.stack([train_data.inverse_transform(y) for y in batch_y])

        condition = [len(np.nonzero(batch_x[x, :, 0])[0]) for x in range(batch_x.shape[0])]
        zero_counter_x += condition  # np.count_nonzero(condition == 0)

        condition = [len(np.nonzero(batch_y[x, :, 0])[0]) for x in range(batch_y.shape[0])]
        zero_counter_y += condition  # np.count_nonzero(condition == 0)

    zero_counter_x = np.array(zero_counter_x)
    print(zero_counter_x)
    print(np.count_nonzero(zero_counter_x == 0))
    zero_counter_x = np.mean(zero_counter_x)
    print(f"{zero_counter_x}")

    zero_counter_y = np.array(zero_counter_y)
    print(zero_counter_y)
    print(np.count_nonzero(zero_counter_y == 0))
    zero_counter_y = np.mean(zero_counter_y)
    print(f"{zero_counter_y}")
    sys.exit(0)

    batch_seq = []

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        plot_batches(batch_x.detach().numpy()[:5])
        sys.exit(0)
        print(f"\tTest {i / len(test_loader)}")

    sys.exit(0)


    val_data, val_loader = data_provider(config, flag='val')
    print("Length: ", len(val_data))

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(val_loader):
        if i % 100 == 0:
            print(f"\tVal {i / len(val_loader)}")




