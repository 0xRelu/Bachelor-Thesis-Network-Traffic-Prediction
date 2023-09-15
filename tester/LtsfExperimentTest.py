import os.path
import sys

import numpy as np
from matplotlib import pyplot as plt

from exp.exp_main import Exp_Main
from models.Config import Config

if __name__ == "__main__":
    etth_config = {
        'iterations': 200,
        'learning_rate': 0.001,
        'lradj': 'type3',

        'data': 'ETTh1',
        'batch_size': 10,
        'freq': "h",
        'root_path': "C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data",
        'data_path': "ETT\\ETTh1.csv",  # "ETT\\ETTh1.csv",  #
        'seq_len': 96,
        'label_len': 40,
        'pred_len': 56,
        'features': "M",
        'target': "M",
        'num_workers': 1,
        'test_flop': False,

        'pct_start': 0.3,
        'use_amp': False,
        'patience': 100,
        'gpu': 0,
        'use_gpu': True,
        'use_multi_gpu': False,

        # transformer
        'model_id': "Transformer",
        'model': "Transformer",
        'embed_type': 4,
        'enc_in': 7,
        'dec_in': 7,
        'c_out': 7,
        'd_model': 1024,  # 512,  # 1024
        'n_heads': 8,  # 8,
        'e_layers': 1,
        'd_layers': 1,
        'd_ff': 4096,  # 2048,  # 4096,  # 8192
        'moving_avg': 25,
        'factor': 3,
        'distil': True,
        'dropout': 0.0,  # 0.05,
        'embed': 'fixed',
        'activation': 'gelu',
        'output_attention': False,
        'do_predict': False,
    }

    cw_config = {
        'iterations': 100,
        'learning_rate': 0.001,
        'lradj': 'type3',

        'data': 'Traffic_Even',
        'batch_size': 10,
        'freq': "h",
        'root_path': "C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data",
        'data_path': "UNI1\\univ1_pt1_even_336_48_96.pkl",  # "ETT\\ETTh1.csv",  #
        'seq_len': 56,
        'label_len': 16,
        'pred_len': 10,
        'features': "M",
        'target': "M",
        'num_workers': 1,
        'test_flop': False,

        'pct_start': 0.3,
        'use_amp': False,
        'patience': 100,
        'gpu': 0,
        'use_gpu': True,
        'use_multi_gpu': False,

        # transformer
        'model_id': "Transformer",
        'model': "Transformer",
        'embed_type': 4,
        'enc_in': 1,
        'dec_in': 1,
        'c_out': 1,
        'd_model': 512,  # 1024
        'n_heads': 8,
        'e_layers': 2,
        'd_layers': 1,
        'd_ff': 2048,  # 4096,  # 8192
        'moving_avg': 25,
        'factor': 3,
        'distil': True,
        'dropout': 0.0, # 0.05,
        'embed': 'fixed',
        'activation': 'gelu',
        'output_attention': False,
        'do_predict': False,
    }

    print("<<<< start >>>>")
    config = Config(etth_config)
    expMain = Exp_Main(config)

    print("[+] Start training loop")
    trues_preds = []
    losses = []

    for n in range(config.iterations):
        loss, t_losses, trues_preds = expMain.train(n)
        losses.append(loss)

    plt.figure()
    plt.plot(np.array(losses), label='loss')
    plt.xlabel('Time')
    plt.ylabel('Size')
    plt.legend()
    plt.savefig(os.path.join("C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\img", "loss.png"))

    y, y_pred = trues_preds[-1]  # trues_pred is list of tensors

    for i in range(y.size(0)):
        y_p, y_pred_p = y[i, :, 0].detach().cpu().numpy(), y_pred[i, :, 0].detach().cpu().numpy()

        plt.figure()
        plt.plot(y_p, label='trues')
        plt.plot(y_pred_p, label='pred')
        plt.xlabel('Time')
        plt.ylabel('Size')
        plt.legend()
        plt.savefig(os.path.join("C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\img", "trues_pred_" + str(i) + ".png"))

    test_dict = expMain.test()
    print(f"test | loss: {test_dict}")
    vali_loss = expMain.vali()
    print(f"val | loss: {vali_loss}")

    sys.exit(0)
