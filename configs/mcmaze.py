from typing import Literal
from dataclasses import dataclass

@dataclass
class McMazeConfig:
    num_channels: int = 137
    num_heldout_channels: int = 45
    mask_ratio: float = 0.90
    pretrain: bool = False
    entire: bool = True


    # Hyperparameters
    scheduler: Literal["cosine", "cosinerestarts"] = "cosine"
    batch_size: int = 64
    learning_rate: float = 5e-4
    weight_decay: float = 5e-4
    warmup_lr: float = 1e-7
    num_epochs: int = 1500
    restart_period: int = 100 # Only used for cosinerestarts.
    num_warmup_epochs: int = 75
    num_workers: int = 8
