from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention
from .layer_scale import LayerScale
from .mlp import Mlp


class Block(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.norm1 = norm_layer(embed_dim)
        self.attn = attn_class(
            embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.ls1 = (
            LayerScale(embed_dim, init_values=init_values)
            if init_values
            else nn.Identity()
        )

        self.norm2 = norm_layer(embed_dim)
        mlp_hidden_embed_dim = int(embed_dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=embed_dim,
            hidden_features=mlp_hidden_embed_dim,
            act_layer=act_layer,
            drop=proj_drop,
            bias=ffn_bias,
        )
        self.ls2 = (
            LayerScale(embed_dim, init_values=init_values)
            if init_values
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DecoderBlock(Block):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            embed_dim,
            num_heads,
            mlp_ratio,
            qkv_bias,
            proj_bias,
            ffn_bias,
            proj_drop,
            attn_drop,
            init_values,
            act_layer,
            norm_layer,
            attn_class,
            ffn_layer,
            *args,
            **kwargs,
        )
        self.norm0 = norm_layer(embed_dim)
        self.attn_0 = Attention(
            embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.ls0 = (
            LayerScale(embed_dim, init_values=init_values)
            if init_values
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        attn_mask: torch.Tensor = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        x = x + self.ls0(self.attn_0(self.norm0(x)))
        x = x + self.ls1(self.attn(self.norm1(x), memory, attn_mask, is_causal))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x
