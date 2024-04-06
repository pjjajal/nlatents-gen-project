import math

import torch
import torch.nn as nn
from torch import Tensor


# This is taken from a pytorch. I have transposed the batch and seq_len.
class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(1, max_len, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, time, embedding_dim]`` or ``[batch_size, channel_patches, time, embedding_dim]``
        """
        if x.dim() == 4:
            B, _, _, e = x.shape
            # (1, max_len, embed_dim) -> (1, 1, max_len, embed_dim)
            # and then add it to the input tensor.
            x = x + self.pe.unsqueeze(1)[:, :, : x.size(2), :] 
            x = x.reshape(B, -1, e)
            return self.dropout(x)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


if __name__ == "__main__":
    pe = PositionalEncoding(6, 0.0)
    y = torch.zeros(1, 2, 6)
    y = pe(y)
    print(y.shape)

    pe = PositionalEncoding(6, 0.0)
    y = torch.zeros(1, 2, 2, 6)
    y = pe(y)
    print(y)
    print(y.shape)
