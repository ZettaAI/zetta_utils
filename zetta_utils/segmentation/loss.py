from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import torch
from torch import nn
from typeguard import typechecked

from zetta_utils import builder


@builder.register("LossWithMask")
@typechecked
class LossWithMask(nn.Module):
    def __init__(
        self,
        criterion: Callable[..., nn.Module],
        reduction: Literal["mean", "sum", "none"] = "sum",
    ) -> None:
        super().__init__()
        try:
            self.criterion = criterion(reduction="none")
        except (KeyError, TypeError):
            self.criterion = criterion()
        assert self.criterion.reduction == "none"
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        trgt: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor | None:
        nmsk = torch.count_nonzero(mask)
        if nmsk.item() == 0:
            return None

        loss = mask * self.criterion(pred, trgt)
        if self.reduction == "none":
            return loss

        loss = torch.sum(loss)

        if self.reduction == "mean":
            assert nmsk.item() > 0
            loss /= nmsk.to(loss.dtype).item()

        return loss


@builder.register("BinaryLossWithMargin")
@typechecked
class BinaryLossWithMargin(LossWithMask):
    def __init__(
        self,
        criterion: Callable[..., nn.Module],
        reduction: Literal["mean", "sum", "none"] = "sum",
        margin: float = 0,
        logits: bool = False,
    ) -> None:
        super().__init__(criterion, reduction)
        self.margin = np.clip(margin, 0, 1)
        self.logits = logits

    def forward(
        self,
        pred: torch.Tensor,
        trgt: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor | None:
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
        margin: float = 0,
    ) -> None:
        super().__init__(criterion, reduction)
        self.margin = np.clip(margin, 0, 1)

    def forward(
        self,
        pred: torch.Tensor,
        trgt: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor | None:
        trgt[torch.eq(trgt, 1)] = 1 - self.margin
        trgt[torch.eq(trgt, 0)] = self.margin
        return super().forward(pred, trgt, mask)
