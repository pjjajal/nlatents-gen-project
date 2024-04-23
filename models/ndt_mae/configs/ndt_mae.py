from dataclasses import dataclass


@dataclass
class EncoderConfig:
    channels: int
    channel_kernel_size: int = 0
    embed_dim: int = 64
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    ffn_bias: bool = True
    proj_bias: bool = True
    proj_drop: float = 0.25
    attn_drop: float = 0.25
    pos_drop: float = 0.20
    init_values: float = 0.1  # layerscale init value, set to 0.1 based on the paper.


@dataclass
class DecoderConfig:
    input_dim: int = 64
    embed_dim: int = 32
    depth: int = 3
    num_heads: int = 8
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    ffn_bias: bool = True
    proj_bias: bool = True
    proj_drop: float = 0.25
    attn_drop: float = 0.25
    init_values: float = 0.1


CONFIGS = {
    "mcmaze_supervised": (EncoderConfig(channels=137), DecoderConfig()),
    "mcmaze_supervised_behav": (EncoderConfig(channels=2), DecoderConfig()),
    "mcmaze_ssl": (
        EncoderConfig(channels=137, embed_dim=128, depth=8, num_heads=16),
        EncoderConfig(channels=2, embed_dim=128, depth=8, num_heads=16),
        DecoderConfig(input_dim=256, embed_dim=64),
    ),
}
