import torch
from torch.nn.utils.rnn import pad_sequence


def padded_collate_fn(batch):
    x_batch, y_batch, x2_batch, y2_batch = zip(*batch)

    padded_x_batch = pad_sequence(x_batch, batch_first=True)
    padded_x2_batch = pad_sequence(x2_batch, batch_first=True)

    return padded_x_batch, torch.stack(y_batch), padded_x2_batch, torch.stack(y2_batch)
