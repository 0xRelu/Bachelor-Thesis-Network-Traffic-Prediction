import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbeddingMicroseconds(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TimeFeatureEmbeddingMicroseconds, self).__init__()

        microseconds_size = 1000
        milliseconds_size = 1000
        second_size = 60
        minute_size = 60
        hour_size = 24
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding

        self.microseconds_embed = Embed(microseconds_size, d_model)
        self.milliseconds_embed = Embed(milliseconds_size, d_model)
        self.second_embed = Embed(second_size, d_model)
        self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):  # x = [year, month, day, hour, minute, second, millisecond, microsecond]
        x = x.long()

        microsecond_x = self.microseconds_embed(x[:, :, 7])
        millisecond_x = self.milliseconds_embed(x[:, :, 6])
        second_x = self.second_embed(x[:, :, 5])
        minute_x = self.minute_embed(x[:, :, 4])
        hour_x = self.hour_embed(x[:, :, 3])
        day_x = self.day_embed(x[:, :, 2])
        month_x = self.month_embed(x[:, :, 1])

        return hour_x + minute_x + second_x + millisecond_x + microsecond_x  # month_x + day_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DirectionEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(DirectionEmbedding, self).__init__()

        direction_amount = 2  # 2 direction

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding

        self.direction_embedding = Embed(direction_amount, d_model)  # (A <-> B) -> (A -> B) | (A <-> B) -> (A <- B)

    def forward(self, x):
        x = x.long()
        return self.direction_embedding(x)


class ProtocolEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(ProtocolEmbedding, self).__init__()

        protocol_amount = 2  # TCP = 0, UDP = 1

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding

        self.protocol_embedding = Embed(protocol_amount, d_model)  # (A <-> B) -> (A -> B) | (A <-> B) -> (A <- B)

    def forward(self, x):
        x = x.long()
        return self.protocol_embedding(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_w_dir_temp(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_w_dir_temp, self).__init__()

        self.value_embedding = nn.Linear(c_in - 1, d_model,
                                         bias=True)  # -1 because we use a separate embedding for direction
        self.direction_embedding = DirectionEmbedding(d_model=d_model)  # assume that direction is the last feature
        self.protocol_embedding = ProtocolEmbedding(d_model=d_model)
        self.temporal_embedding = TimeFeatureEmbeddingMicroseconds(d_model=d_model, embed_type=embed_type,
                                                                   freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):  # x = [B,L,2] ~ (size,direction,protocol), x_mark= [B,L,2] ~ (milliseconds, microseconds)
        x = self.value_embedding(x[:, :, :-2]) + self.direction_embedding(x[:, :, -2]) \
            + self.protocol_embedding(x[:, :, -1]) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class DataEmbedding_w_temp(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_w_temp, self).__init__()

        self.value_embedding = nn.Linear(c_in, d_model)  # -1 because we use a separate embedding for direction
        self.temporal_embedding = TimeFeatureEmbeddingMicroseconds(d_model=d_model, embed_type=embed_type,
                                                                   freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):  # x = [B,L,c_in], x_mark= [B,L,2] ~ (milliseconds, microseconds)
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)  # TODO add directional embedding
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class DataEmbedding_wo_pos_temp(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos_temp, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_temp(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_temp, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
