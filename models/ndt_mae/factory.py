import os
from pathlib import Path
from omegaconf import OmegaConf
from .configs.ndt_mae import EncoderConfig, DecoderConfig
from .encoder import Encoder
from .decoder import Decoder
from .mae import MAE

def mae_factory(encoder_conf: EncoderConfig, decoder_conf: DecoderConfig):
    encoder_conf = OmegaConf.structured(encoder_conf)
    decoder_conf = OmegaConf.structured(decoder_conf)
    model = MAE(encoder_conf, decoder_conf)
    return model

if __name__ == "__main__":
    mae = mae_factory(EncoderConfig(channels=137), decoder_conf=DecoderConfig())
    import torch
    x = torch.randn(1, 140, 137)
    mae(x)