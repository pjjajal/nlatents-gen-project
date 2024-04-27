import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import List

import h5py
import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from lightning.pytorch.callbacks import LearningRateMonitor
from nlb_tools.evaluation import evaluate
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader

from configs.mcmaze import TRAIN_CONFIGS, McMazeConfig, McMazeConfigCLIPesque
from dataloaders.mcmaze import McMazeDataset
from models.ndt_mae.configs.ndt_mae import CONFIGS, DecoderConfig, EncoderConfig
from models.ndt_mae.decoder import Decoder, HeldoutDecoder
from models.ndt_mae.encoder import Encoder
from utils.schedulers import (
    CosineAnnealingWithWarmup,
    WarmRestartsCosineAnnealingWithWarmUp,
)

torch.set_float32_matmul_precision("medium")


class LightningNdtMae(L.LightningModule):
    def __init__(
        self,
        spike_encoder_conf: EncoderConfig,
        behavior_encoder_conf: EncoderConfig,
        decoder_conf: DecoderConfig,
        config: McMazeConfig | McMazeConfigCLIPesque,
        behaviour: bool = False,
        rnn: bool = False,
    ):
        super().__init__()
        self.behaviour = behaviour
        self.rnn = rnn

        spike_encoder_conf = OmegaConf.structured(spike_encoder_conf)
        behavior_encoder_conf = OmegaConf.structured(behavior_encoder_conf)
        decoder_conf = OmegaConf.structured(decoder_conf)

        self.config = config
        self.best_perf = 0
        self.current_loss = 0.0

        # This is the common encoder for held-in neurons
        self.spike_encoder = Encoder(**spike_encoder_conf)
        self.behavior_encoder = Encoder(**behavior_encoder_conf)

        # Pretraining Stuff
        # Contrastive Loss
        self.proj_spike = nn.Linear(
            spike_encoder_conf["embed_dim"], decoder_conf["embed_dim"]
        )
        self.proj_behavior = nn.Linear(
            behavior_encoder_conf["embed_dim"], decoder_conf["embed_dim"]
        )
        self.proj_spike_heldout = nn.Linear(
            spike_encoder_conf["embed_dim"], decoder_conf["embed_dim"]
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_heldout = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_heldin = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Fine-tuning Stuff
        # This is the decoder for held-in neurons and predicts the masked time-steps
        if rnn:
            self.forward_decoder = nn.LSTM(
                input_size=(
                    behavior_encoder_conf.embed_dim
                    if behaviour
                    else spike_encoder_conf.embed_dim
                ),
                hidden_size=decoder_conf.embed_dim,
                num_layers=6,
                batch_first=True,
                dropout=0.2,
            )
        else:
            self.forward_decoder = Decoder(
                input_dim=(
                    behavior_encoder_conf.embed_dim
                    if behaviour
                    else spike_encoder_conf.embed_dim
                ),
                **decoder_conf,
            )
        # This is the decoder for held-out neurons and predicts held out neurons at
        # the unmasked time-steps
        if rnn:
            self.held_out_decoder = nn.LSTM(
                input_size=(
                    behavior_encoder_conf.embed_dim
                    if behaviour
                    else spike_encoder_conf.embed_dim
                ),
                hidden_size=decoder_conf.embed_dim,
                num_layers=6,
                batch_first=True,
                dropout=0.2,
            )
        else:
            self.held_out_decoder = Decoder(
                input_dim=(
                    behavior_encoder_conf.embed_dim
                    if behaviour
                    else spike_encoder_conf.embed_dim
                ),
                **decoder_conf,
            )

        # This is the projection layer for the forward decoder
        self.norm = nn.LayerNorm(decoder_conf["embed_dim"])
        self.norm_heldout = nn.LayerNorm(decoder_conf["embed_dim"])
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

        if not self.config.pretrain:
            if self.behaviour:
                latents, mask, ids_restore, ids_keep = self.behavior_encoder.forward(
                    train_behavior, mask_ratio
                )
            else:
                latents, mask, ids_restore, ids_keep = self.spike_encoder.forward(
                    train_spikes_heldin, mask_ratio
                )
            if self.rnn:
                pred_heldin, _ = self.forward_decoder.forward(latents)
                pred_heldout, _ = self.held_out_decoder.forward(latents)
            else:
                pred_heldin = self.forward_decoder.forward(latents, ids_restore)
                pred_heldout = self.held_out_decoder.forward(latents, ids_restore)

            pred_heldin = self.proj(self.norm(pred_heldin))
            pred_heldout = self.proj_heldout(self.norm_heldout(pred_heldout))

            loss_heldin = self.classifier(pred_heldin, train_spikes_heldin)
            loss_heldin = loss_heldin.mean()

            loss_heldout = self.classifier(pred_heldout, train_spikes_heldout)
            loss_heldout = loss_heldout.mean()

            self.train_rates_heldin.append(torch.exp(pred_heldin).detach().cpu())
            self.train_rates_heldout.append(torch.exp(pred_heldout).detach().cpu())

            loss = loss_heldin + loss_heldout
            self.log("train_loss_heldin", loss_heldin, on_epoch=True, on_step=False)
            self.log("train_loss_heldout", loss_heldout, on_epoch=True, on_step=False)
            self.log("train_loss", loss, on_epoch=True, prog_bar=True, on_step=False)
            return loss
        else:
            pred_spike, mask, ids_restore, ids_keep = self.spike_encoder.forward(
                train_spikes_heldin, self.config.mask_ratio_cross
            )
            pred_behavior, mask, ids_restore, ids_keep = self.behavior_encoder.forward(
                train_behavior, self.config.mask_ratio_cross
            )

            # Contrastive Loss -- Behaviour to Spike
            pred_spike_clip = pred_spike.mean(dim=1)
            pred_behavior_clip = pred_behavior.mean(dim=1)

            pred_spike_clip = self.proj_spike(pred_spike_clip)
            pred_behavior_clip = self.proj_behavior(pred_behavior_clip)

            # normalized features
            pred_spike_clip = F.normalize(pred_spike_clip, dim=-1)
            pred_behavior_clip = F.normalize(pred_behavior_clip, dim=-1)

            # cosine similarity as logits
            logit_scale = self.logit_scale.exp()
            logits_per_spike = pred_spike_clip @ pred_behavior_clip.T
            logits_per_behaviour = logits_per_spike.T

            logits_per_spike = logits_per_spike * logit_scale
            logits_per_behaviour = logits_per_behaviour * logit_scale

            labels = torch.arange(
                logits_per_spike.shape[0],
                device=logits_per_spike.device,
                dtype=torch.long,
            )
            contrastive_loss = (
                F.cross_entropy(logits_per_spike, labels)
                + F.cross_entropy(logits_per_behaviour, labels)
            ) / 2

            # Spike Contrative Loss
            pred_spike_1, mask, ids_restore, ids_keep = self.spike_encoder.forward(
                train_spikes_heldin, mask_ratio
            )
            pred_spike_2, mask, ids_restore, ids_keep = self.spike_encoder.forward(
                train_spikes_heldin, mask_ratio
            )

            pred_spike_1 = pred_spike_1.mean(dim=1)
            pred_spike_2 = pred_spike_2.mean(dim=1)

            pred_spike_1 = self.proj_spike(pred_spike_1)
            pred_spike_2 = self.proj_spike(pred_spike_2)

            # normalized features
            pred_spike_1 = F.normalize(pred_spike_1, dim=-1)
            pred_spike_2 = F.normalize(pred_spike_2, dim=-1)

            # cosine similarity as logits
            logit_scale = self.logit_scale_heldin.exp()
            logits_per_spike_1 = pred_spike_1 @ pred_spike_2.T
            logits_per_spike_2 = logits_per_spike_1.T

            logits_per_spike_1 = logits_per_spike_1 * logit_scale
            logits_per_spike_2 = logits_per_spike_2 * logit_scale

            labels = torch.arange(
                logits_per_spike_1.shape[0],
                device=logits_per_spike_1.device,
                dtype=torch.long,
            )
            contrastive_loss_spike = (
                F.cross_entropy(logits_per_spike_1, labels)
                + F.cross_entropy(logits_per_spike_2, labels)
            ) / 2

            loss = contrastive_loss + contrastive_loss_spike
            self.current_loss = loss

            self.log("pretrain_loss", loss, on_epoch=True, prog_bar=True, on_step=False)
            self.log(
                "contrastive_loss",
                contrastive_loss,
                on_epoch=True,
                prog_bar=True,
                on_step=False,
            )
            self.log(
                "contrastive_loss_spike",
                contrastive_loss_spike,
                on_epoch=True,
                prog_bar=True,
                on_step=False,
            )
            return loss

    def on_train_epoch_end(self) -> None:
        if self.config.pretrain:
            if self.best_perf < self.current_loss:
                self.best_perf = self.current_loss
                torch.save(self.state_dict(), "./model_clip_best_pretrained.pth")

    def on_validation_epoch_start(self) -> None:
        self.eval_rates_heldin = []
        self.eval_rates_heldout = []

    def validation_step(self, batch, batch_idx):
        eval_behavior, eval_spikes_heldin, eval_spikes_heldout = batch

        if self.behaviour:
            latents, mask, ids_restore, ids_keep = self.behavior_encoder.forward(
                eval_behavior, 0
            )
        else:
            latents, mask, ids_restore, ids_keep = self.spike_encoder.forward(
                eval_spikes_heldin, 0
            )

        if self.rnn:
            pred_heldin, _ = self.forward_decoder.forward(latents)
            pred_heldout, _ = self.held_out_decoder.forward(latents)
        else:
            pred_heldin = self.forward_decoder.forward(latents, ids_restore)
            pred_heldout = self.held_out_decoder.forward(latents, ids_restore)

        pred_heldin = self.proj(self.norm(pred_heldin))
        pred_heldout = self.proj_heldout(self.norm_heldout(pred_heldout))

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

        if results[0]["mc_maze_split"]["co-bps"] > self.best_perf:
            save_path = f"./model_best"
            if self.behaviour:
                save_path += "_behaviour"
            save_path += ".pth"
            torch.save(self.state_dict(), save_path)

    def configure_optimizers(self):
        parameters = list()

        # Pretraining
        if self.config.pretrain:
            parameters += list(self.spike_encoder.parameters())
            parameters += list(self.behavior_encoder.parameters())
            parameters += list(self.proj_spike.parameters())
            parameters += list(self.proj_behavior.parameters())
            parameters += list(self.proj_spike_heldout.parameters())
            parameters.append(self.logit_scale)
            parameters.append(self.logit_scale_heldin)
            parameters.append(self.logit_scale_heldout)

        # Finetuning
        else:
            parameters += list(self.forward_decoder.parameters())
            parameters += list(self.held_out_decoder.parameters())
            parameters += list(self.proj.parameters())
            parameters += list(self.proj_heldout.parameters())
            parameters += list(self.norm.parameters())
            parameters += list(self.norm_heldout.parameters())
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
    parser = argparse.ArgumentParser(
        "Training script for supervised training of Transformer on McMaze. You can train to predict spike from spikes and spikes from behavior, using the correct flags."
    )
    parser.add_argument(
        "--behaviour",
        action="store_true",
        help="Train to predict spikes from behavior.",
        default=False,
    )
    parser.add_argument(
        "--rnn", action="store_true", help="Use RNN decoder of Transformer."
    )
    parser.add_argument(
        "--pretrain", action="store_true", help="Pretrain the model.", default=False
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    spike_encoder_conf, behaviour_encoder_conf, decoder_conf = CONFIGS["mcmaze_ssl"]
    if args.pretrain:
        config = TRAIN_CONFIGS["mcmaze_ssl_pt"]
    elif args.rnn:
        config = TRAIN_CONFIGS["mcmaze_ssl_rnn"]
    else:
        config = TRAIN_CONFIGS["mcmaze_ssl"]
    config.pretrain = args.pretrain

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

    model = LightningNdtMae(
        spike_encoder_conf,
        behaviour_encoder_conf,
        decoder_conf,
        config,
        args.behaviour,
        args.rnn,
    )

    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint), strict=False)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = L.Trainer(
        max_epochs=config.num_epochs,
        benchmark=True,
        precision="bf16-mixed",
        enable_checkpointing=False,  # Disable automatic checkpointing (we do this manually).
        callbacks=[lr_monitor],
        num_sanity_val_steps=0,
        check_val_every_n_epoch=50,
        gradient_clip_val=1.0,
        limit_val_batches=0.0 if args.pretrain else 1.0,
    )
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
