from __future__ import annotations

from typing import Callable, Literal, Tuple

import numpy as np
import torch
from torch import nn
from typeguard import typechecked

from zetta_utils import builder


@builder.register("LossWithMask")
@typechecked
class LossWithMask(nn.Module):  # pragma: no cover
    def __init__(
        self,
        criterion: Callable[..., nn.Module],
        reduction: Literal["mean", "sum", "none"] = "sum",
        balancer: nn.Module | None = None,
        mean_reduction_clip: float = 0.1,
        return_loss_map: bool = False,
    ) -> None:
        """
        :param mean_reduction_clip:
            Clip multiplier when computing mean loss when loss mask is partial, i.e.,
            multipler = sum / max(mask_ratio, clip).
        """
        super().__init__()
        try:
            self.criterion = criterion(reduction="none")
        except (KeyError, TypeError):
            self.criterion = criterion()
        assert self.criterion.reduction == "none"
        self.reduction = reduction
        self.balancer = balancer
        self.balanced = False
        assert 0.0 < mean_reduction_clip <= 1.0
        self.mean_reduction_clip = mean_reduction_clip
        self.return_loss_map = return_loss_map

    def forward(
        self,
        pred: torch.Tensor,
        trgt: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:

        # Optional class balancing
        if (not self.balanced) and (self.balancer is not None):
            mask = self.balancer(trgt, mask)

        loss_map = mask * self.criterion(pred, trgt)
        if self.reduction == "none":
            return loss_map

        loss = torch.sum(loss_map)

        if self.reduction == "mean":
            nmsk = torch.clip(
                torch.count_nonzero(mask),
                min=mask.numel() * self.mean_reduction_clip,
            )
            loss /= nmsk.to(loss.dtype).item()

        if self.return_loss_map:
            return loss, loss_map
        else:
            return loss


@builder.register("BinaryLossWithMargin")
@typechecked
class BinaryLossWithMargin(LossWithMask):
    def __init__(
        self,
        criterion: Callable[..., nn.Module],
        reduction: Literal["mean", "sum", "none"] = "sum",
        balancer: nn.Module | None = None,
        margin: float = 0,
        logits: bool = False,
    ) -> None:
        super().__init__(criterion, reduction, balancer)
        self.margin = np.clip(margin, 0, 1)
        self.logits = logits

    def forward(
        self,
        pred: torch.Tensor,
        trgt: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        # Optional class balancing
        if self.balancer is not None:
            mask = self.balancer(trgt, mask)
            self.balanced = True

        high = 1 - self.margin
        low = self.margin
        activ = torch.sigmoid(pred) if self.logits else pred
        hmask = torch.ge(activ, high) * torch.eq(trgt, 1)
        lmask = torch.le(activ, low) * torch.eq(trgt, 0)
        mask *= 1 - (hmask | lmask).to(mask.dtype)
        return super().forward(pred, trgt, mask)


@builder.register("BinaryLossWithInverseMargin")
@typechecked
class BinaryLossWithInverseMargin(LossWithMask):
    def __init__(
        self,
        criterion: Callable[..., nn.Module],
        reduction: Literal["mean", "sum", "none"] = "sum",
        balancer: nn.Module | None = None,
        margin: float = 0,
    ) -> None:
        super().__init__(criterion, reduction, balancer)
        self.margin = np.clip(margin, 0, 1)

    def forward(
        self,
        pred: torch.Tensor,
        trgt: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        # Optional class balancing
        if self.balancer is not None:
            mask = self.balancer(trgt, mask)
            self.balanced = True

        trgt[torch.eq(trgt, 1)] = 1 - self.margin
        trgt[torch.eq(trgt, 0)] = self.margin
        return super().forward(pred, trgt, mask)
