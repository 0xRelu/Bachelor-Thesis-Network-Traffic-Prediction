import torch
import torch.nn as nn
import torch.optim as optim


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(configs.seq_len, configs.d_model)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(configs.d_model, configs.pred_len)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.permute(0, 2, 1)
        return x
