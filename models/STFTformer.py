import copy

import torch
import torch.nn as nn
from models import Transformer, RLinear
from utils.tools import dotdict
import torch.nn.functional as F

class Model(nn.Module):
    """
    """

    def __init__(self, configs, model: nn.Module = RLinear.Model):
        super(Model, self).__init__()
        # set enc in and enc out
        self.seg_len = configs.seg_len
        self.overlap = configs.overlap

        self.pred_len = configs.pred_len

        _configs = dotdict(dict(configs))  # copy that it does not impact
        _configs.seq_len = 1 + _configs.seq_len // self.overlap
        _configs.pred_len = 1 + _configs.pred_len // self.overlap

        self.model = model(_configs)

    def _stft(self, x):  # expect x = (B,L,1)
        x = x.squeeze()  # x = (B, L)
        x = torch.stft(x, n_fft=self.seg_len, hop_length=self.overlap, center=True,
                       pad_mode="reflect", normalized=True,
                       return_complex=True, onesided=True)  # x = (B, self.seg_len + 1, 1 + L // hop_length)
        x = x.permute(0, 2, 1)  # x = (B, 1 + L // hop_length, self.seg_len + 1)
        x = torch.cat((x.real, x.imag), dim=2)
        return x

    def _istft(self, x):  # x = (B, 1 + L // hop_length, self.seg_len + 1)
        x = torch.complex(x[:, :, :x.shape[2] // 2], x[:, :, x.shape[2] // 2:])
        x = x.permute(0, 2, 1)  # x = (B, self.seg_len + 1, 1 + L // hop_length)
        x = torch.istft(x, n_fft=self.seg_len, hop_length=self.overlap, center=True,
                        normalized=True)  # x = (B, L)
        x = F.pad(x, (0, self.pred_len - x.shape[1]), mode='constant', value=0)
        x = x.unsqueeze(-1)
        return x

    def create_time(self, x):
        pass

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):  # x_enc = (B,L,1), x_dec = (B,L2,1)
        x_enc = self._stft(x_enc)  # x_enc = (B, 1 + L // hop_length, self.seg_len + 1)
        x_dec = self._stft(x_dec)  # x_dec = (B, 1 + L2 // hop_length, self.seg_len + 1)

        x_out = self.model(x_enc, x_dec)  # , x_mark_enc, x_dec, x_mark_dec)  # x_out = (B, 1 + L2 // hop_length, self.seg_len + 1)
        x_out = self._istft(x_out)  # x_out = (B,L2,1)
        return x_out
