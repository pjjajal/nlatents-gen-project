from dataclasses import dataclass

@dataclass
class EncoderConfig:
    channels: int
    channel_kernel_size: int = 0
    embed_dim: int = 256
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    ffn_bias: bool = True
    proj_bias: bool = True
    proj_drop: float = 0.4
    attn_drop: float = 0.4
    pos_drop: float = 0.2
    init_values: float = 0.1 # layerscale init value, set to 0.1 based on the paper.

@dataclass
class DecoderConfig:
    input_dim: int = 256
    embed_dim: int = 64
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    ffn_bias: bool = True
    proj_bias: bool = True
    proj_drop: float = 0.4
    attn_drop: float = 0.4
    init_values: float = 0.1