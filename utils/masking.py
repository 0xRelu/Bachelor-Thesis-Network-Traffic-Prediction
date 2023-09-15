import torch
from torch import Tensor


class PaddingMask:
    def __init__(self, batch_x):  # x: [B, L, F]
        pad_token_index = 0
        # batch_mask = (batch_x == pad_token_index)  # transform to false/true if equals 0
        # batch_mask = torch.all(batch_mask, dim=2)  # [B, L]
        # batch_mask = torch.einsum("bl,bs->bls", batch_mask, batch_mask)  # [B, L, L]
        # self._mask = torch.unsqueeze(batch_mask, 1)  # [B, 1, L, L]
        #
        # batch_mask = (batch_x == pad_token_index)
        # batch_mask = torch.all(batch_mask, dim=2)
        # batch_mask = batch_mask.unsqueeze(1).unsqueeze(1)
        # num_heads = 8
        # batch_mask = batch_mask.repeat(1, num_heads, batch_x.size(1), 1)
        # self._mask2 = batch_mask

        batch_mask = (batch_x == pad_token_index)  # transform to false/true if equals 0
        batch_mask = torch.all(batch_mask, dim=2)  # [B, L]
        self._mask3 = batch_mask

    @property
    def mask(self):
        return self._mask3


def generate_square_subsequent_mask(dim1: int, dim2: int) -> Tensor:
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)


class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask:
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask
