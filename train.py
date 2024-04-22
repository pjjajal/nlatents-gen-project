import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import List

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from lightning.pytorch.callbacks import LearningRateMonitor
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader

from models.ndt_mae.configs.ndt_mae import EncoderConfig, DecoderConfig, CONFIGS
from models.ndt_mae.encoder import Encoder
from models.ndt_mae.decoder import Decoder, HeldoutDecoder
from configs.mcmaze import McMazeConfig, TRAIN_CONFIGS
from dataloaders.mcmaze import McMazeDataset
from utils.schedulers import (
    CosineAnnealingWithWarmup,
    WarmRestartsCosineAnnealingWithWarmUp,
)
from nlb_tools.evaluation import evaluate
import h5py

torch.set_float32_matmul_precision("medium")


class LightningNdtMae(L.LightningModule):
    def __init__(
        self,
        encoder_conf: EncoderConfig,
        decoder_conf: DecoderConfig,
        config: McMazeConfig,
        behaviour: bool = False,
    ):
        super().__init__()
        self.behaviour = behaviour

        encoder_conf = OmegaConf.structured(encoder_conf)
        decoder_conf = OmegaConf.structured(decoder_conf)

        self.config = config
        self.best_perf = 0

        # This is the common encoder for held-in neurons
        self.encoder = Encoder(**encoder_conf)

        # This is the decoder for held-in neurons and predicts the masked time-steps
        self.forward_decoder = Decoder(**decoder_conf)
        # This is the decoder for held-out neurons and predicts held out neurons at
        # the unmasked time-steps
        self.held_out_decoder = Decoder(**decoder_conf)

        # This is the projection layer for the forward decoder
        self.proj = nn.Linear(decoder_conf["embed_dim"], config.num_channels)
        self.proj_heldout = nn.Linear(
            decoder_conf["embed_dim"], config.num_heldout_channels
        )

        self.classifier = nn.PoissonNLLLoss(reduction="none", log_input=True)

    def on_train_epoch_start(self) -> None:
        self.train_rates_heldin = []
        self.train_rates_heldout = []

    def training_step(self, batch, batch_idx):
        train_behavior, train_spikes_heldin, train_spikes_heldout = batch

        mask_ratio = self.config.mask_ratio

        if self.behaviour:
            inputs = train_behavior
        latents, mask, ids_restore, ids_keep = self.encoder.forward(
            inputs, mask_ratio
        )

        pred_heldin = self.forward_decoder.forward(latents, ids_restore)
        pred_heldout = self.held_out_decoder.forward(latents, ids_restore)

        pred_heldin = self.proj(pred_heldin)
        pred_heldout = self.proj_heldout(pred_heldout)

        loss_heldin = self.classifier(pred_heldin, train_spikes_heldin)
        if self.config.pretrain:
            loss_heldin = (loss_heldin * mask.unsqueeze(-1)).sum() / mask.sum()
        else:
            loss_heldin = loss_heldin.mean()

        loss_heldout = self.classifier(pred_heldout, train_spikes_heldout)
        if self.config.pretrain:
            loss_heldout = (loss_heldout * (~mask.bool()).unsqueeze(-1)).sum() / (
                (~mask.bool()).sum()
            )
        else:
            loss_heldout = loss_heldout.mean()


        self.train_rates_heldin.append(torch.exp(pred_heldin).detach().cpu())
        self.train_rates_heldout.append(torch.exp(pred_heldout).detach().cpu())

        loss = loss_heldin + loss_heldout
        self.log("train_loss_heldin", loss_heldin, on_epoch=True, on_step=False)
        self.log("train_loss_heldout", loss_heldout, on_epoch=True, on_step=False)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, on_step=False)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.eval_rates_heldin = []
        self.eval_rates_heldout = []

    def validation_step(self, batch, batch_idx):
        eval_behavior, eval_spikes_heldin, eval_spikes_heldout = batch
        
        if self.behaviour:
            inputs = eval_behavior
        latents, mask, ids_restore, ids_keep = self.encoder.forward(
            inputs, 0.0
        )

        pred_heldin = self.forward_decoder.forward(latents, ids_restore)
        pred_heldout = self.held_out_decoder.forward(latents, ids_restore)

        pred_heldin = self.proj(pred_heldin)
        pred_heldout = self.proj_heldout(pred_heldout)

        loss_heldin = self.classifier(pred_heldin, eval_spikes_heldin).mean()
        loss_heldout = self.classifier(pred_heldout, eval_spikes_heldout).mean()
        loss = loss_heldin + loss_heldout

        self.log("eval_loss", loss, on_epoch=True, prog_bar=True, on_step=False)
        self.eval_rates_heldin.append(torch.exp(pred_heldin).detach().cpu())
        self.eval_rates_heldout.append(torch.exp(pred_heldout).detach().cpu())

    def on_validation_epoch_end(self) -> None:
        preds = {
            "mc_maze": {
                "eval_rates_heldin": torch.cat(self.eval_rates_heldin).numpy(),
                "eval_rates_heldout": torch.cat(self.eval_rates_heldout).numpy(),
                "train_rates_heldin": torch.cat(self.train_rates_heldin).numpy(),
                "train_rates_heldout": torch.cat(self.train_rates_heldout).numpy(),
            }
        }

        with h5py.File("datasets/mcmaze_val_target_bw5.h5", "r") as f:
            target_dict = {}
            for key in f["mc_maze"]:
                target_dict[key] = f["mc_maze"][key][()]
        targets = {"mc_maze": target_dict}

        results = evaluate(targets, preds)
        print(results)
        self.log_dict(results[0]["mc_maze_split"])

        if self.config.pretrain:
            if results[0]["mc_maze_split"]["co-bps"] > self.best_perf:
                torch.save(self.encoder.state_dict(), "./encoder_best.pth")
        else:
            if results[0]["mc_maze_split"]["co-bps"] > self.best_perf:
                save_path = f"./model_best"
                if self.behaviour:
                    save_path += "_behaviour"
                save_path += ".pth"
                torch.save(self.state_dict(), save_path)

    def configure_optimizers(self):
        parameters = list()
        parameters += list(self.encoder.parameters())
        parameters += list(self.forward_decoder.parameters())
        parameters += list(self.held_out_decoder.parameters())
        parameters += list(self.proj.parameters())
        parameters += list(self.proj_heldout.parameters())
        optimizer = optim.AdamW(
            parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        if self.config.scheduler == "cosine":
            scheduler = CosineAnnealingWithWarmup(
                optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.warmup_lr,
                warmup_epochs=self.config.num_warmup_epochs,
            )
        elif self.config.scheduler == "cosinerestarts":
            scheduler = WarmRestartsCosineAnnealingWithWarmUp(
                optimizer,
                T_0=self.config.restart_period,
                T_mult=1,
                eta_min=self.config.warmup_lr,
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training script for supervised training of Transformer on McMaze. You can train to predict spike from spikes and spikes from behavior, using the correct flags.")
    parser.add_argument("--behaviour", action="store_true", help="Train to predict spikes from behavior.", default=False)
    args = parser.parse_args()

    if args.behaviour:
        encoder_conf, decoder_conf = CONFIGS["mcmaze_supervised_behav"]
        config = TRAIN_CONFIGS["mcmaze_supervised_behaviour"]
    else:
        encoder_conf, decoder_conf = CONFIGS["mcmaze_supervised"]
        config = TRAIN_CONFIGS["mcmaze_supervised"]
        # encoder_conf = EncoderConfig(channels=137)
        # decoder_conf = DecoderConfig()

    train_dataset = McMazeDataset("datasets/mcmaze_train_bw5.h5")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    val_dataset = McMazeDataset("datasets/mcmaze_val_bw5.h5", split="eval")
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = LightningNdtMae(encoder_conf, decoder_conf, config, args.behaviour)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = L.Trainer(
        max_epochs=config.num_epochs,
        benchmark=True,
        precision="bf16-mixed",
        enable_checkpointing=False,  # Disable automatic checkpointing (we do this manually).
        callbacks=[lr_monitor],
        num_sanity_val_steps=0,
        check_val_every_n_epoch=50,
        gradient_clip_val=1.0
    )
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
