from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint
from .layers import (
    Block,
    Attention,
    Mlp,
    LayerScale,
)


class Encoder(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        proj_drop=0.0,
        attn_drop=0.0,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=None,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        norm_layer = partial(nn.LayerNorm, eps=1e-6)


        blocks = [Block(
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
        ) for _ in range(depth)]