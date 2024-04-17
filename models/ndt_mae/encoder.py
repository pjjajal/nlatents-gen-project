from functools import partial
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

    # This is also taken from the MAE repo. 
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        if not self.training or mask_ratio <= 0:
            return x, None, None, None

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep


    def forward(self, x , mask_ratio=0.0):
        x = self.embed(x)
        x = self.pos_embed(x)
        x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)
        
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x, mask, ids_restore, ids_keep

if __name__ == "__main__":
    encoder = Encoder()
    x = torch.randn(1, 10, 137)
    x, mask, ids_restore, ids_keep = encoder(x, mask_ratio=0.1)
    print(x.shape, mask.shape, ids_restore.shape, ids_keep.shape)