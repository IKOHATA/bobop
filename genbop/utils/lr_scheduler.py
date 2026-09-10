
from __future__ import annotations

import math
import types
import warnings
from bisect import bisect_right
from collections import Counter
from functools import partial, wraps
from typing import (
    Any,
    Callable,
    cast,
    Literal,
    Optional,
    SupportsFloat,
    TYPE_CHECKING,
    TypedDict,
    Union,
)
from typing_extensions import override, Self
from weakref import ref

from torch import inf, Tensor

from torch.optim.optimizer import Optimizer

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

from torch.optim.lr_scheduler import LRScheduler

class SpikeAwareReduceLROnPlateau(LRScheduler):

    def __init__(
        self,
        optimizer: Optimizer,
        mode: Literal["min", "max"] = "min",
        factor: float = 0.1,
        patience: int = 10,
        spike_tolerance: float = 0.1,
        threshold: float = 1e-4,
        threshold_mode: Literal["rel", "abs"] = "rel",
        cooldown: int = 0,
        min_lr: Union[list[float], float] = 0,
        eps: float = 1e-8,
    ):  # noqa: D107
        if factor >= 1.0:
            raise ValueError("Factor should be < 1.0.")
        self.factor = factor

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        self.optimizer = optimizer

        if isinstance(min_lr, (list, tuple)):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(
                    f"expected {len(optimizer.param_groups)} min_lrs, got {len(min_lr)}"
                )
            self.default_min_lr = None
            self.min_lrs = list(min_lr)
        else:
            self.default_min_lr = min_lr
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.spike_tolerance = spike_tolerance
        self.cooldown = cooldown
        self.eps = eps
        self.last_epoch = 0
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
        self._init_is_better(
            mode=mode, threshold=threshold, threshold_mode=threshold_mode
        )
        self._reset()

    def _reset(self):
        """Reset num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0
        self.spike = 0

    def step(self, metrics: SupportsFloat, epoch=None) -> None:  # type: ignore[override]
        """Perform a step."""
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            if self.is_spike(current, self.best):
                self.num_bad_epochs = 0
                self.spike = 1
            else:
                self.num_bad_epochs += 1
                self.spike = 0

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def _reduce_lr(self, epoch):
        if len(self.optimizer.param_groups) != len(self.min_lrs):
            if self.default_min_lr is None:
                raise RuntimeError(
                    "The number of param groups in the `optimizer` "
                    f"({len(self.optimizer.param_groups)}) differs "
                    f"from when `ReduceLROnPlateau` was initialized "
                    f"({len(self.min_lrs)}), usually due to a new "
                    "param group being added to the optimizer. Please "
                    "modify the `min_lrs` field to match the length "
                    "of the `optimizer` param groups."
                )
            else:
                self.min_lrs = [self.default_min_lr] * len(self.optimizer.param_groups)

        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group["lr"])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group["lr"] = new_lr

    @property
    def in_cooldown(self):  # noqa: D102
        return self.cooldown_counter > 0

    def is_better(self, a, best):  # noqa: D102
        if self.mode == "min" and self.threshold_mode == "rel":
            rel_epsilon = 1.0 - self.threshold
            return a < best * rel_epsilon

        elif self.mode == "min" and self.threshold_mode == "abs":
            return a < best - self.threshold

        elif self.mode == "max" and self.threshold_mode == "rel":
            rel_epsilon = self.threshold + 1.0
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def is_spike(self, a, best):  # noqa: D102
        if self.mode == "min":
            rel_epsilon = 1.0 + self.spike_tolerance
            return a > best * rel_epsilon 

        elif self.mode == "max":
            rel_epsilon = 1.0 - self.spike_tolerance
            return a < best * rel_epsilon

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError("threshold mode " + threshold_mode + " is unknown!")

        # the worse value for the chosen mode
        if mode == "min":
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the scheduler's state."""
        self.__dict__.update(state_dict)
        self._init_is_better(
            mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode
        )