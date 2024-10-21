from typing import Optional, List
from bisect import bisect_right
from math import cos, pi
from torch.optim.lr_scheduler import _LRScheduler


class LRSchedulerWithWarmup(_LRScheduler):
    def __init__(
        self,
        optimizer,
        mode: str,
        warmup_epochs: int = 10,
        warmup_method: str = "linear",
        total_epochs: int = 100,
        n_iter_per_epoch: int = 100,  # Number of iterations per epoch
        last_epoch: int = -1,
        milestones: Optional[List[int]] = [3, 5],
        gamma: Optional[float] = 0.1,
        warmup_factor: Optional[float] = 0.1,
        start_lr: Optional[float] = 1e-6,
        end_lr: Optional[float] = 5e-6,
        power: Optional[float] = 1.0,
    ):
        """
        LRSchedulerWithWarmup with iteration-based learning rate scheduling.
        """
        if mode not in ("step", "exp", "poly", "cosine", "linear"):
            raise ValueError(
                f"Invalid mode {mode}. Choose from 'step', 'exp', 'poly', 'cosine', 'linear'"
            )
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                f"Invalid warmup_method {warmup_method}. Choose 'constant' or 'linear'"
            )

        self.mode = mode
        self.warmup_epochs = warmup_epochs
        self.warmup_method = warmup_method
        self.total_epochs = total_epochs
        self.n_iter_per_epoch = n_iter_per_epoch  # Added for iteration granularity

        # Convert total_epochs and warmup_epochs to total iterations
        self.total_iters = total_epochs * n_iter_per_epoch
        self.warmup_iters = warmup_epochs * n_iter_per_epoch

        # Step mode specific checks
        if mode == "step":
            if milestones is None or gamma is None:
                raise ValueError(
                    "'milestones' and 'gamma' are required for 'step' mode"
                )
            if not list(milestones) == sorted(milestones):
                raise ValueError(
                    f"Milestones should be increasing integers. Got {milestones}"
                )
            self.milestones = [
                m * n_iter_per_epoch for m in milestones
            ]  # Convert milestones to iteration scale
            self.gamma = gamma

        elif mode in ("poly", "cosine"):
            if start_lr is None or end_lr is None:
                raise ValueError(
                    f"'start_lr' and 'end_lr' are required for '{mode}' mode"
                )
            self.start_lr = start_lr
            self.end_lr = end_lr

        if mode == "poly" and power is None:
            raise ValueError("'power' is required for 'poly' mode")

        self.warmup_factor = warmup_factor
        self.power = power

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Compute the learning rate for the current iteration based on the mode and warmup settings.
        """
        current_iter = self.last_epoch  # Treat `last_epoch` as the iteration number

        # Warmup phase
        if current_iter < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
                return [base_lr * warmup_factor for base_lr in self.base_lrs]

            elif self.warmup_method == "linear":
                alpha = current_iter / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
                return [
                    # self.start_lr + alpha * (base_lr - self.start_lr)
                    self.start_lr + (base_lr - self.start_lr) * warmup_factor
                    for base_lr in self.base_lrs
                ]

        # Post-warmup phase
        iter_ratio = (current_iter - self.warmup_iters) / (
            self.total_iters - self.warmup_iters
        )

        if self.mode == "step":
            return [
                base_lr * self.gamma ** bisect_right(self.milestones, current_iter)
                for base_lr in self.base_lrs
            ]
        elif self.mode == "exp":
            return [base_lr * (1 - iter_ratio) for base_lr in self.base_lrs]
        elif self.mode == "linear":
            return [base_lr * (1 - iter_ratio) for base_lr in self.base_lrs]
        elif self.mode == "poly":
            factor = (1 - iter_ratio) ** self.power
            return [
                self.end_lr + (base_lr - self.end_lr) * factor
                for base_lr in self.base_lrs
            ]
        elif self.mode == "cosine":
            factor = 0.5 * (1 + cos(pi * iter_ratio))
            return [
                self.end_lr + (base_lr - self.end_lr) * factor
                for base_lr in self.base_lrs
            ]
        raise NotImplementedError
