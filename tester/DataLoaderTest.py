import sys

from data_provider.data_factory import data_provider
from utils.tools import dotdict

if __name__ == "__main__":
    print("Start")

    cw_config = {
        'data': 'Traffic_Single',
        'batch_size': 128,
        'freq': "h",
        'root_path': "C:\\Users\\nicol\\PycharmProjects\\BA_LTSF_w_Transformer\\data",
        'data_path': "UNI1\\univ1_pt1_single_336_48_12.pkl",
        'seq_len': 336,
        'label_len': 48,
        'pred_len': 12,
        'features': "M",
        'target': "M",
        'num_workers': 1,
        'embed': 'timeF',
    }

    config = dotdict(cw_config)

    train_data, train_loader = data_provider(config, flag='train')  #, collate_fn=padded_collate_fn)
    print("Length: ", len(train_data))

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        if i % 10 == 0:
            print(f"Train {i / len(train_loader)} | Shape_x: {batch_x.shape} - {batch_y.shape} ")

    train_data, train_loader = data_provider(config, flag='test')
    print("Length: ", len(train_data))
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        if i % 100 == 0:
            print(f"Test {i / len(train_loader)}")

    train_data, train_loader = data_provider(config, flag='val')
    print("Length: ", len(train_data))
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        if i % 100 == 0:
            print(f"Val {i / len(train_loader)}")

    sys.exit(0)
