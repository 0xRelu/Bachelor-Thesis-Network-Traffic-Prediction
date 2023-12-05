import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):
    """
    Just one Linear layer
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.Linear = nn.Linear(self.seq_len, self.pred_len)  # doesn't matter - just to make sure it works :)

    def forward(self, x, y):  # x = [B,L,S], y = [B,P,S]
        x = torch.mean(x, dim=1, keepdim=True)  # x = [B,]
        x = x.expand(-1, y.shape[1], y.shape[2])  # x = [B,P,S]
        return x
