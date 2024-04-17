# I wrote this myself. It is a modified version of the CosineAnnealingLR scheduler in PyTorch.
import torch
import torch.optim as optim
from typing import List, Tuple
import math
import warnings


class CosineAnnealingWithWarmup(optim.lr_scheduler.CosineAnnealingLR):
    def __init__(
        self,
        optimizer: optim.Optimizer,
        T_max: int,
        eta_min: float = 0,
        last_epoch: int = -1,
        verbose: bool = "deprecated",
        warmup_epochs: int = 10,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        T_max = T_max - warmup_epochs  # These are the annealing epochs.
        super().__init__(optimizer, T_max, eta_min, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_epochs:
            return self._warmup_lr(self.last_epoch)
        else:
            return self._cosine_lr(self.last_epoch - self.warmup_epochs)

    # This is nearly-identical to PyTorch implementation, but last epoch is now an argument.
    def _warmup_lr(self, last_epoch) -> List[float]:
        if last_epoch == 0:
            return [self.eta_min for group in self.optimizer.param_groups]
        return [
            self.eta_min + last_epoch * (base_lr - self.eta_min) / self.warmup_epochs
            for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
        ]

    # This is nearly-identical to PyTorch implementation, but last epoch is now an argument.
    def _cosine_lr(self, last_epoch) -> List[float]:
        if last_epoch == 0:
            return [base_lr for base_lr in self.base_lrs]
        elif self._step_count == 1 and last_epoch > 0:
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (1 + math.cos((last_epoch) * math.pi / self.T_max))
                / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif (last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [
            (1 + math.cos(math.pi * last_epoch / self.T_max))
            / (1 + math.cos(math.pi * (last_epoch - 1) / self.T_max))
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]


class WarmRestartsCosineAnnealingWithWarmUp(optim.lr_scheduler.CosineAnnealingWarmRestarts):
    def __init__(
        self,
        optimizer: optim.Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0,
        last_epoch: int = -1,
        verbose: bool = "deprecated",
        warmup_epochs: int = 10,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        T_0 = T_0 - warmup_epochs  # These are the annealing epochs.
        super().__init__(optimizer, T_0, T_mult, eta_min, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_epochs:
            return self._warmup_lr(self.last_epoch)
        else:
            return self._cosine_lr(self.T_cur - self.warmup_epochs)

    # This is nearly-identical to PyTorch implementation, but last epoch is now an argument.
    def _warmup_lr(self, last_epoch) -> List[float]:
        if last_epoch == 0:
            return [self.eta_min for group in self.optimizer.param_groups]
        return [
            self.eta_min + last_epoch * (base_lr - self.eta_min) / self.warmup_epochs
            for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
        ]
    
    def _cosine_lr(self, curr):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
            
        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * curr / self.T_i)) / 2
                for base_lr in self.base_lrs]
    
