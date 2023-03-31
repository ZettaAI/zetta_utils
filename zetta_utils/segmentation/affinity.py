from __future__ import annotations

from typing import Callable, Literal, Sequence

import torch
from torch import nn
from typeguard import typechecked

from zetta_utils import builder, tensor_ops

from .loss import LossWithMask


@typechecked
class EdgeSampler:
    def __init__(self, edges: Sequence[Sequence[int]]) -> None:
        assert len(edges) > 0
        assert all(len(edge) == 3 for edge in edges)
        self.edges = list(edges)

    def generate_edges(self) -> Sequence[Sequence[int]]:
        return list(self.edges)


@typechecked
class EdgeDecoder(nn.Module):
    def __init__(self, edges: Sequence[Sequence[int]]) -> None:
        super().__init__()
        assert len(edges) > 0
        assert all(len(edge) == 3 for edge in edges)
        self.edges = list(edges)

    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        assert x.ndim >= 4
        num_channels = x.shape[-4]  # CZYX
        assert num_channels == len(self.edges)
        assert 0 <= idx < num_channels
        data = x[..., [idx], :, :, :]
        edge = self.edges[idx]
        return tensor_ops.get_disp_pair(data, edge)[1]


@typechecked
class EdgeCRF(nn.Module):
    def __init__(
        self,
        criterion: Callable[..., nn.Module],
        reduction: Literal["mean", "sum", "none"],
    ) -> None:
        super().__init__()
        try:
            criterion_ = criterion(reduction="none")
        except (KeyError, TypeError):
            criterion_ = criterion()
        assert criterion_.reduction == "none"
        if isinstance(criterion_, LossWithMask):
            self.criterion = criterion_
        else:
            self.criterion = LossWithMask(criterion, reduction="none")
        self.reduction = reduction

    def forward(
        self,
        preds: Sequence[torch.Tensor],
        trgts: Sequence[torch.Tensor],
        masks: Sequence[torch.Tensor],
    ) -> torch.Tensor | list[torch.Tensor] | None:
        assert len(preds) == len(trgts) == len(masks) > 0

        dtype = preds[0].dtype
        device = preds[0].device

        losses = []
        nmsk = torch.tensor(0, dtype=dtype, device=device)

        for pred, trgt, mask in zip(preds, trgts, masks):
            # TODO: class rebalancing
            loss_ = self.criterion(pred, trgt, mask)
            if loss_ is not None:
                losses.append(loss_)
            nmsk += torch.count_nonzero(mask)

        if nmsk.item() == 0:
            assert len(losses) == 0
            return None

        if self.reduction == "none":
            return losses

        # Sum up losses
        losses = list(map(torch.sum, losses))
        loss = torch.sum(torch.stack(losses))

        if self.reduction == "mean":
            assert nmsk.item() > 0
            loss /= nmsk.to(loss.dtype).item()

        return loss


@builder.register("AffinityLoss")
@typechecked
class AffinityLoss(nn.Module):
    def __init__(
        self,
        edges: Sequence[Sequence[int]],
        criterion: Callable[..., nn.Module],
        reduction: Literal["mean", "sum", "none"] = "none",
    ) -> None:
        super().__init__()
        self.sampler = EdgeSampler(edges)
        self.decoder = EdgeDecoder(edges)
        self.criterion = EdgeCRF(criterion, reduction)

    def forward(
        self,
        pred: torch.Tensor,
        trgt: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor | list[torch.Tensor] | None:
        preds = []  # type: list[torch.Tensor]
        trgts = []
        masks = []
        edges = self.sampler.generate_edges()
        for idx, edge in enumerate(edges):
            affmap, affmsk = tensor_ops.seg_to_aff(trgt, edge, mask=mask)
            preds.append(self.decoder(pred, idx))
            trgts.append(affmap)
            masks.append(affmsk)
        return self.criterion(preds, trgts, masks)
