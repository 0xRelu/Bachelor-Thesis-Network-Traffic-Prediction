import torch
from torch import nn, Tensor

from layers.Embed import DataEmbedding_w_dir_temp, DataEmbedding_w_temp
from utils.masking import generate_square_subsequent_mask, PaddingMask


class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """

    def __init__(self, configs):
        super().__init__()

        self.configs = configs

        if configs.embed_type == 5:
            self.enc_embedding = DataEmbedding_w_temp(configs.enc_in, configs.d_model, configs.embed,
                                                      configs.freq,
                                                      configs.dropout)
            self.dec_embedding = DataEmbedding_w_temp(configs.dec_in, configs.d_model, configs.embed,
                                                      configs.freq,
                                                      configs.dropout)
        elif configs.embed_type == 6:
            self.enc_embedding = DataEmbedding_w_dir_temp(configs.enc_in, configs.d_model, configs.embed,
                                                          configs.freq,
                                                          configs.dropout)
            self.dec_embedding = DataEmbedding_w_temp(configs.dec_in, configs.d_model, configs.embed,
                                                      configs.freq,
                                                      configs.dropout)  # we don't have direction in target sequence
        else:
            raise AttributeError("Illegal embed_type" + configs.embed_type)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=configs.d_model,
            nhead=configs.n_heads,
            dim_feedforward=configs.d_ff,
            dropout=configs.dropout,
            batch_first=True,
            activation=configs.activation
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=configs.e_layers,
            norm=torch.nn.LayerNorm(configs.d_model),
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=configs.d_model,
            nhead=configs.n_heads,
            dim_feedforward=configs.d_model,
            dropout=configs.dropout,
            batch_first=True,
            activation=configs.activation
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=configs.d_layers,
            norm=torch.nn.LayerNorm(configs.d_model)
        )

        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forward(self, src: Tensor, src_mark: Tensor, tgt: Tensor, tgt_mark: Tensor,
                src_mask=None, tgt_mask=None) -> Tensor:
        """
        Returns a tensor of shape:
        [target_sequence_length, batch_size, num_predicted_features]

        Args:
            src: the encoder's output sequence. Shape: (S,E) for unbatched input,
                 (S, N, E) if batch_first=False or (N, S, E) if
                 batch_first=True, where S is the source sequence length,
                 N is the batch size, and E is the number of features (1 if univariate)
            src_mark: TODO
            tgt: the sequence to the decoder. Shape: (T,E) for unbatched input,
                 (T, N, E)(T,N,E) if batch_first=False or (N, T, E) if
                 batch_first=True, where T is the target sequence length,
                 N is the batch size, and E is the number of features (1 if univariate)
            tgt_mark: TODO
            src_mask: the mask for the src sequence to prevent the model from
                      using data points from the target sequence
            tgt_mask: the mask for the tgt sequence to prevent the model from
                      using data points from the target sequence
        """

        if src_mask is None:
            src_mask = PaddingMask(src).mask

        src = self.enc_embedding(src, src_mark)
        src = self.encoder(src=src, src_key_padding_mask=src_mask)

        if tgt_mask is None:
            tgt_mask = generate_square_subsequent_mask(tgt.size(1), tgt.size(1)).to(src.device)

        decoder_output = self.dec_embedding(tgt, tgt_mark)
        decoder_output = self.decoder(
            tgt=decoder_output,
            memory=src,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_mask
        )
        decoder_output = self.projection(decoder_output)

        return decoder_output
