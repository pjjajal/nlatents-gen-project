import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # Attn dropout
        self.attn_drop_prob = attn_drop

        # Proj dropout
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop_prob, scale=self.scale
        ).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        # Attn dropout
        self.attn_drop_prob = attn_drop

        # Proj dropout
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        attn_mask: Tensor = None,
        is_causal: bool = False,
    ) -> Tensor:
        B, N, C = x.shape
        B_mem, N_mem, C_mem = memory.shape

        q = self.scale * self.q(x).reshape(
            B, N, self.num_heads, C // self.num_heads
        ).permute(0, 2, 1, 3)
        kv = (
            self.kv(memory)
            .reshape(B_mem, N_mem, 2, self.num_heads, C_mem // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        k, v = kv[0], kv[1]

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop_prob,
            scale=self.scale,
            attn_mask=attn_mask,
            is_causal=is_causal,
        ).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
