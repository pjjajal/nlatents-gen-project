from typing import Literal
from dataclasses import dataclass


@dataclass
class McMazeConfig:
    num_channels: int = 137
    num_heldout_channels: int = 45
    mask_ratio: float = 0.40
    pretrain: bool = False
    entire: bool = True

    # Hyperparameters
    scheduler: Literal["cosine", "cosinerestarts"] = "cosine"
    batch_size: int = 64
    learning_rate: float = 7.5e-2
    weight_decay: float = 5e-2
    warmup_lr: float = 1e-7
    num_epochs: int = 750
    restart_period: int = 100  # Only used for cosinerestarts.
    num_warmup_epochs: int = 75
    num_workers: int = 8


@dataclass
class McMazeConfigCLIPesque:
    num_channels: int = 137
    num_heldout_channels: int = 45
    mask_ratio: float = 0.40
    mask_ratio_cross: float = 0.40
    pretrain: bool = False
    entire: bool = True

    # Hyperparameters
    scheduler: Literal["cosine", "cosinerestarts"] = "cosine"
    batch_size: int = 128
    learning_rate: float = 1.0e-2
    weight_decay: float = 5e-2
    warmup_lr: float = 1e-7
    num_epochs: int = 750
    restart_period: int = 100  # Only used for cosinerestarts.
    num_warmup_epochs: int = 75
    num_workers: int = 8


TRAIN_CONFIGS = {
    "mcmaze_supervised": McMazeConfig(),
    "mcmaze_supervised_behaviour": McMazeConfig(
        mask_ratio=0.45,
        learning_rate=7.5e-2,
    ),
    "mcmaze_ssl_pt": McMazeConfigCLIPesque(
        mask_ratio=0.50,
        mask_ratio_cross=0.40,
        batch_size=512,
        learning_rate=3e-2,
        weight_decay=5e-2,
        num_epochs=750,
    ),
    "mcmaze_ssl": McMazeConfigCLIPesque(
        mask_ratio=0.00,
        batch_size=32,
        learning_rate=3.5e-3,
    ),
    "mcmaze_ssl_rnn": McMazeConfigCLIPesque(
        mask_ratio=0.00,
        batch_size=32,
        learning_rate=7.5e-3,
    ),
}
