import torch
import torch.nn as nn
from torch.nn import functional as F
from layers.Invertible import RevIN


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.amplitude = 1.0
        self.frequency = 0.5
        self.phase = 0.0
        self.offset = nn.Parameter(torch.tensor(0.0))

        self.pred_len = configs.pred_len

    def forward(self, x):
        y = self.amplitude * torch.sin(torch.linspace(0, 2 * torch.pi, self.pred_len).to(x.device)) + self.offset
        y = y.unsqueeze(-1)
        y = torch.stack([y] * x.shape[0])
        return y
