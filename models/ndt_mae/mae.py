import torch
import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder

class MAE(nn.Module):
    def __init__(self, encoder_conf, decoder_conf):
        super().__init__()
        self.encoder = Encoder(**encoder_conf)
        self.decoder = Decoder(**decoder_conf)
        self.classifier = nn.PoissonNLLLoss(reduction='none', log_input=True)

    def forward(self, x, mask_ratio = 0.1):
        latents, mask, ids_restore = self.encoder.forward(x, mask_ratio)
        preds = self.decoder.forward(latents, ids_restore)
        loss = self.classifier(preds, x)
        loss = (loss * mask).sum() / mask.sum()
        return loss
    

if __name__ == "__main__":
    model = MAE()
    x = torch.randn(1, 140, 137)
    model(x)