import torch
import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder

class MAE(nn.Module):
    def __init__(self, encoder_conf, decoder_conf):
        super().__init__()
        self.encoder = Encoder(**encoder_conf)
        self.decoder = Decoder(**decoder_conf)
        self.proj = nn.Linear(decoder_conf['embed_dim'], 137)
        self.classifier = nn.PoissonNLLLoss(reduction='none', log_input=True)

    def forward(self, x, mask_ratio = 0.1):
        latents, mask, ids_restore = self.encoder.forward(x, mask_ratio)
        print(latents.shape, mask.shape, ids_restore.shape)
        preds = self.decoder.forward(latents, ids_restore)
        preds = self.proj(preds)
        loss = self.classifier(preds, x)
        loss = (loss * mask.unsqueeze(-1)).sum() / mask.sum()
        return loss
    

if __name__ == "__main__":
    model = MAE()
    x = torch.randn(1, 140, 137)
    model(x)