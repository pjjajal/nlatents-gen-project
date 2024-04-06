from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
from .layers import Block, Attention, Mlp, LayerScale, PositionalEncoding, Embed


class Encoder(nn.Module):
    def __init__(
        self,
        channels: int = 137,  # This is the default value for MC_MAZE.
        channel_kernel_size: int = None,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        proj_drop=0.0,
        attn_drop=0.0,
        pos_drop=0.0,  # positional encoding dropout
        init_values=None,  # for layerscale: None or 0 => no layerscale
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.embed = Embed(
            channels,
            embed_dim,
            channel_kernel_size,
        )
        self.pos_embed = PositionalEncoding(embed_dim, pos_drop)
        blocks = [
            Block(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop=proj_drop,
                attn_drop=attn_drop,
                norm_layer=norm_layer,
                init_values=init_values,
                ffn_layer=Mlp,
            )
            for _ in range(depth)
        ]

        self.blocks = nn.ModuleList(blocks)
        self.norm = norm_layer(embed_dim)

        self.apply(self._init_weights)

    # This was taken from the MAE repo.
    # Link: https://github.com/facebookresearch/mae/tree/main
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.embed(x)
        x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x
