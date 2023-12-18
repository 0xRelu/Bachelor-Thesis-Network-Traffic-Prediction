import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import amp

from data_provider.data_factory import data_provider
from utils.metrics import MSE
from utils.tools import dotdict


def plot_batches(batch_x: np.ndarray):
    for y in range(batch_x.shape[0]):
        plt.plot(batch_x[y], label=y)
        plt.legend()
        plt.show()

def plot_stft(batch_x, f, t):
    assert len(batch_x.shape) == 3

    for i in range(len(batch_x)):
        trans_batch = batch_x[i, :, :6] + 1j * batch_x[i, :, 6:]
        trans_batch = trans_batch.numpy().transpose()

        t_i = t[i][:trans_batch.shape[1]]

        plt.pcolormesh(t_i, f[i], np.abs(trans_batch), shading='auto')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        plt.title('Spectrogram')
        plt.colorbar(label='?? [dB]')
        plt.show()


if __name__ == "__main__":
    print("Start")

    cw_config = {
        'data': 'Traffic_Even',
        'batch_size': 128,
        'freq': "h",
        'root_path': "C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data\\UNI1_n",
        'data_path': "univ1_pt1_even4_1000.pkl",  # univ1_pt1_even_336_48_12_1000.pkl
        'seq_len': 336,
        'label_len': 100,
        'pred_len': 96,
        'features': "M",
        'target': "bytes",
        'num_workers': 1,
        'embed': 'timeF',
        'transform': None,
        # 'smooth_param': '(100, 90)',  # 12
        'seq_stride': 100
    }

    config = dotdict(cw_config)

    train_data, train_loader = data_provider(config, flag='test')  # , collate_fn=padded_collate_fn)
    print("Length: ", len(train_data))

    counter = 0

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        for j, seq in enumerate(batch_x):
            seq = train_data.inverse_transform(seq)
            pred = train_data.inverse_transform(batch_y[j])

            non_zero_count = np.count_nonzero(seq)
            percentage_non_zero = non_zero_count / seq.shape[0]

            if percentage_non_zero >= 0.5:
                counter += 1
                plt.plot(seq, label=counter)
                plt.legend()
                plt.show()

                print([seq.squeeze().tolist(), pred.squeeze().tolist()])

                plt.plot(pred, label=counter)
                plt.legend()
                plt.show()

            if counter > 10:
                break

        if counter > 10:
            break

        if i % 100 == 0:
            print(f"\tTrain {i / len(train_loader)}")

    train_data, train_loader = data_provider(config, flag='val')  # , collate_fn=padded_collate_fn)
    print("Length: ", len(train_data))

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        if i % 100 == 0:
            print(f"\tVal {i / len(train_loader)}")

    train_data, train_loader = data_provider(config, flag='test')  # , collate_fn=padded_collate_fn)
    print("Length: ", len(train_data))

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        if i % 100 == 0:
            print(f"\tVal {i / len(train_loader)}")

    sys.exit(0)
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

    val_data, val_loader = data_provider(config, flag='val')
    print("Length: ", len(val_data))

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(val_loader):
        if i % 100 == 0:
            print(f"\tVal {i / len(val_loader)}")
