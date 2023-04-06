from __future__ import annotations

from functools import partial
from typing import Callable, Literal, Sequence, cast

import attrs
import numpy as np
import torch
from torch import nn
from typeguard import typechecked

from zetta_utils import builder, tensor_ops
from zetta_utils.geometry import Vec3D
from zetta_utils.geometry.bbox import Slices3D
from zetta_utils.layer import JointIndexDataProcessor
from zetta_utils.layer.volumetric import VolumetricIndex

from .loss import LossWithMask

NDIM = 3


@typechecked
class EdgeSampler:
    def __init__(self, edges: Sequence[Sequence[int]]) -> None:
        assert len(edges) > 0
        assert all(len(edge) == NDIM for edge in edges)
        self.edges = list(edges)

    def generate_edges(self) -> Sequence[Sequence[int]]:
        return list(self.edges)


@typechecked
class EdgeDecoder(nn.Module):
    def __init__(self, edges: Sequence[Sequence[int]], pad_crop: bool) -> None:
        super().__init__()
        assert len(edges) > 0
        assert all(len(edge) == NDIM for edge in edges)
        self.edges = list(edges)
        self.pad_crop = pad_crop

    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        assert x.ndim >= 4
        num_channels = x.shape[-(NDIM + 1)]  # CZYX
        assert num_channels == len(self.edges)
        assert 0 <= idx < num_channels
        data = x[..., [idx], :, :, :]
        if not self.pad_crop:
            edge = self.edges[idx]
            data = tensor_ops.get_disp_pair(data, edge)[1]
        return data


@typechecked
class EdgeCRF(nn.Module):
    def __init__(
        self,
        criterion: Callable[..., nn.Module],
        reduction: Literal["mean", "sum", "none"],
        balancer: nn.Module | None,
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
        self.balancer = balancer

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
            if self.balancer is not None:
                mask = self.balancer(trgt, mask)
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


@typechecked
def _compute_slices(
    edges: Sequence[Sequence[int]],
    pad_crop: bool,
) -> list[Slices3D]:
    assert len(edges) > 0
    assert all(len(edge) == NDIM for edge in edges)

    if not pad_crop:
        slices = cast(Slices3D, tuple([slice(0, None)] * NDIM))
        return [slices] * len(edges)

    # Padding in the negative & positive directions
    pad_neg = -np.amin(np.array(edges), axis=0, initial=0)
    pad_pos = np.amax(np.array(edges), axis=0, initial=0)

    # Compute slices for each edge
    result = []
    for edge in edges:
        slices_ = []
        for lpad, rpad, disp in zip(pad_neg, pad_pos, edge):
            start, end = lpad, -rpad
            if disp > 0:
                end += disp
            else:
                start += disp
            slices_.append(slice(start, None) if end == 0 else slice(start, end))
        slices = cast(Slices3D, tuple(slices_))
        result.append(slices)

    return result


@typechecked
def _expand_slices_dim(slices: Slices3D, ndim: int) -> tuple[slice, ...]:
    extra_dims = ndim - NDIM
    assert extra_dims >= 0
    extra_slc = [slice(0, None)] * extra_dims
    result = tuple(extra_slc + list(slices))
    return result


@builder.register("AffinityLoss")
@typechecked
class AffinityLoss(nn.Module):
    def __init__(
        self,
        edges: Sequence[Sequence[int]],
        criterion: Callable[..., nn.Module],
        reduction: Literal["mean", "sum", "none"] = "none",
        balancer: nn.Module | None = None,
        pad_crop: bool = False,
    ) -> None:
        super().__init__()
        self.slices = _compute_slices(edges, pad_crop)
        self.sampler = EdgeSampler(edges)
        self.decoder = EdgeDecoder(edges, pad_crop)
        self.criterion = EdgeCRF(criterion, reduction, balancer)

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
        for idx, (edge, slices) in enumerate(zip(edges, self.slices)):
            aff, msk = tensor_ops.seg_to_aff(trgt[slices], edge, mask=mask)
            preds.append(self.decoder(pred, idx))
            trgts.append(aff)
            masks.append(msk)
        return self.criterion(preds, trgts, masks)


@builder.register("AffinityProcessor")
@typechecked
@attrs.mutable
class AffinityProcessor(JointIndexDataProcessor):  # pragma: no cover
    source: str
    spec: dict[str, Sequence[Sequence[int]]]

    pad_neg: Vec3D[int] = attrs.field(init=False)
    pad_pos: Vec3D[int] = attrs.field(init=False)
    slices: list[Slices3D] = attrs.field(init=False)
    prepared_resolution: Vec3D | None = attrs.field(init=False, default=None)

    def __attrs_post_init__(self):
        edges_union = []
        for edges in self.spec.values():
            assert len(edges) > 0
            assert all(len(edge) == NDIM for edge in edges)
            edges_union += list(edges)

        # Padding in the negative & positive directions
        self.pad_neg = Vec3D(*np.amin(np.array(edges_union), axis=0, initial=0))
        self.pad_pos = Vec3D(*np.amax(np.array(edges_union), axis=0, initial=0))

        # Slices for cropping
        self.slices = _compute_slices(edges_union, pad_crop=True)

    def _crop_data(self, data: torch.Tensor) -> torch.Tensor:
        assert self.prepared_resolution is not None
        idx = (
            VolumetricIndex.from_coords(
                start_coord=Vec3D(0, 0, 0),
                end_coord=Vec3D(*data.shape[-3:]),
                resolution=Vec3D(*self.prepared_resolution),
            )
            .translated_start(-self.pad_neg)
            .translated_end(-self.pad_pos)
        )
        slc = _expand_slices_dim(idx.to_slices(), ndim=data.ndim)
        return data[slc]

    def process_index(
        self, idx: VolumetricIndex, mode: Literal["read", "write"]
    ) -> VolumetricIndex:
        """Pad index to have sufficient context for computing affinities"""
        self.prepared_resolution = idx.resolution
        result = idx.translated_start(self.pad_neg).translated_end(self.pad_pos)
        return result

    def process_data(
        self, data: dict[str, torch.Tensor], mode: Literal["read", "write"]
    ) -> dict[str, torch.Tensor]:
        # Process other data first
        for key, value in data.items():
            if not key.startswith(self.source):
                data[key] = self._crop_data(value)

        # Segmentation and mask
        seg = data[self.source]
        mask = data[self.source + "_mask"]

        # Expand slices dim
        slices = list(map(partial(_expand_slices_dim, ndim=seg.ndim), self.slices))

        # Process affinity
        for target, edges in self.spec.items():
            affs, msks = [], []
            for edge, slc in zip(edges, slices[: len(edges)]):
                aff, msk = tensor_ops.seg_to_aff(seg[slc], edge, mask=mask[slc])
                affs.append(aff)
                msks.append(msk)
            data[target] = torch.cat(affs, dim=-4)
            data[target + "_mask"] = torch.cat(msks, dim=-4)
            slices = slices[len(edges) :]

        # Process segmentation and mask
        data[self.source] = self._crop_data(seg)
        data[self.source + "_mask"] = self._crop_data(mask)

        self.prepared_resolution = None
        return data
