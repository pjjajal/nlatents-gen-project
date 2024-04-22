from functools import partial
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import (
    Block,
    DecoderBlock,
    Attention,
    CrossAttention,
    Mlp,
    LayerScale,
    PositionalEncoding,
    Embed,
)


class Decoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
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
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.embed = nn.Linear(input_dim, embed_dim)

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        blocks = [
            Block(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                norm_layer=norm_layer,
                init_values=init_values,
                ffn_layer=Mlp,
            )
            for _ in range(depth)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.norm = norm_layer(embed_dim)

        # Initialize
        torch.nn.init.normal_(self.mask_token, std=0.02)
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

    def forward(self, x, ids_restore):
        x = self.embed(x)

        # append mask tokens to sequence
        if ids_restore is not None:
            num_mask_tokens = ids_restore.shape[1] + 1 - x.shape[1]
            mask_tokens = self.mask_token.repeat(x.shape[0], num_mask_tokens, 1)
            x = torch.cat([x, mask_tokens], dim=1)
            x = torch.gather(
                x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
            )  # unshuffle

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x


class HeldoutDecoder(Decoder):
    def forward(self, x, ids_restore):
        x = self.embed(x)

        mask_tokens = self.mask_token.repeat(x.shape[0], 1, 1)
        x = torch.cat([x, mask_tokens], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        # print(x.shape, mask_tokens.shape, x[:, mask_tokens.shape[1]:, :].shape)
        return x[:, mask_tokens.shape[1] :, :]


class AutoRegressiveDecoderFaster(Decoder):
    def __init__(
        self,
        input_dim: int = 768,
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
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.embed = nn.Linear(input_dim, embed_dim)

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        blocks = [
            DecoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                norm_layer=norm_layer,
                init_values=init_values,
                ffn_layer=Mlp,
                attn_class=CrossAttention,
            )
            for _ in range(depth)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.norm = norm_layer(embed_dim)

        # Initialize
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def forward(self, x, ids_restore):
        if ids_restore is not None:
            seqlen = ids_restore.shape[1]
        else:
            seqlen = x.shape[1]

        x = self.embed(x)

        input_tokens = torch.zeros(x.shape[0], seqlen, self.embed_dim, device=x.device)
        # Attention Masking
        mask = torch.full(
            (1, input_tokens.shape[1], input_tokens.shape[1]),
            float("-inf"),
            device=x.device,
        )
        mask = torch.triu(mask, diagonal=1)
        for block in self.blocks:
            input_tokens = block.forward(input_tokens, memory=x, attn_mask=None, is_causal=False)
        input_tokens = self.norm(input_tokens)
        return input_tokens

