import torch
from torch.nn.utils.rnn import pad_sequence

from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, \
    Dataset_Traffic_Singe_Packets, Dataset_Traffic_Even, Dataset_Test, Dataset_Traffic_Even_n
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Traffic_Single': Dataset_Traffic_Singe_Packets,
    'Traffic_Even': Dataset_Traffic_Even,
    'Traffic_Even_N': Dataset_Traffic_Even_n,
    'Test': Dataset_Test,
    'custom': Dataset_Custom,
}


def data_provider(args, flag, collate_fn=None):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        random_seed=args.random_seed,
        transform=args.transform,
        smooth_param=args.smooth_param
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn=collate_fn
    )
    return data_set, data_loader
