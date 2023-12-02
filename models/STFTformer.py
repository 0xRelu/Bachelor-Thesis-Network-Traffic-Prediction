import copy

import torch
import torch.nn as nn
from models import Transformer, RLinear, Informer, DLinear, NLinear, Linear, PatchTST
from models.ns_models import ns_Transformer
from utils.tools import dotdict
import torch.nn.functional as F


class Model(nn.Module):
    """
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        # set enc in and enc out
        self.seg_len = configs.seg_len
        self.hop_len = configs.hop_len
        self.pad = configs.pad
        self.model_name = configs.model_name

        self.seq_len = configs.seg_len
        self.pred_len = configs.pred_len

        self.model = self.get_model(configs)

    def get_model(self, configs):
        _configs = dotdict(dict(configs))  # copy dict

        if not self.pad:
            assert (_configs.seq_len - self.seg_len) % self.hop_len == 0
            assert (_configs.pred_len - self.seg_len) % self.hop_len == 0

        _configs.seq_len = 1 + (_configs.seq_len - self.seg_len) // self.hop_len
        _configs.pred_len = 1 + (_configs.pred_len - self.seg_len) // self.hop_len

        model_dict = {
            'Transformer': Transformer,
            'Informer': Informer,
            'Non-Stationary': ns_Transformer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'RLinear': RLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
        }
        return model_dict[self.model_name].Model(_configs).float()

    def _stft(self, x):  # expect x = (B,L,1)
        x = x.squeeze()  # x = (B, L)
        x = torch.stft(x, n_fft=self.seg_len, hop_length=self.hop_len, center=False,
                       pad_mode="reflect", normalized=True,
                       return_complex=True, onesided=True)  # x = (B, (n_fft // 2) + 1, 1 + (L - n_fft) // hop_length
        x = x.permute(0, 2, 1)  # x = (B, 1 + L // hop_length, (n_fft // 2) + )
        x = torch.cat((x.real, x.imag), dim=2)
        return x

    def _istft(self, x):  # x = (B, 1 + L // hop_length, self.seg_len + 1)
        x = torch.complex(x[:, :, :x.shape[2] // 2], x[:, :, x.shape[2] // 2:])
        x = x.permute(0, 2, 1)  # x = (B, self.seg_len + 1, 1 + L // hop_length)
        x = torch.istft(x, n_fft=self.seg_len, hop_length=self.hop_len, center=False,
                        normalized=True)  # x = (B, L)
        if self.pad:
            x = F.pad(x, (0, self.pred_len - x.shape[1]), mode='constant', value=0)
        x = x.unsqueeze(-1)
        return x

    def create_time(self, x_mark):
        max_ = 1 + (x_mark.shape[1] - self.seg_len) // self.hop_len
        return x_mark[:, :max_:self.hop_len]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):  # x_enc = (B,L,1), x_dec = (B,L2,1)
        x_enc = self._stft(x_enc)  # x_enc = (B, 1 + L // hop_length, self.seg_len + 1)
        x_dec = self._stft(x_dec)  # x_dec = (B, 1 + L2 // hop_length, self.seg_len + 1)

        x_mark_enc = self.create_time(x_mark_enc)
        x_mark_dec = self.create_time(x_mark_dec)

        def _run_model():
            if 'RLinear' in self.model_name:
                outputs = self.model(x_enc, x_dec[:, -self.pred_len:])
            elif 'Linear' in self.model_name or 'TST' in self.model_name:
                outputs = self.model(x_enc)
            else:
                outputs = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return outputs

        x_out = _run_model()  # x_out = (B, 1 + L2 // hop_length, self.seg_len + 1)
        x_out = self._istft(x_out)  # x_out = (B,L2,1)
        return x_out
