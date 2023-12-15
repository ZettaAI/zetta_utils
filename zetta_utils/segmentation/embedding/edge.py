from __future__ import annotations

import random
from functools import partial
from typing import Literal, Sequence

import torch
from torch import nn
from typeguard import typechecked

from zetta_utils import builder, tensor_ops
from zetta_utils.geometry import Vec3D

from ..affinity import EdgeCRF
from ..loss import LossWithMask

NDIM = 3


@typechecked
class EdgeSampler:
    def __init__(
        self,
        edges: Sequence[Sequence[int]] | None,
        bounds: Sequence[Sequence[int]],
    ) -> None:
        edges = [] if edges is None else edges
        assert len(edges) > 0 or len(bounds) > 0
        assert all(len(edge) == NDIM for edge in edges)
        assert all(len(bound) == NDIM for bound in bounds)
        assert all(bound != (0, 0, 0) for bound in bounds)
        self.edges = list(edges)
        self.bounds = list(bounds)

    def generate_edges(
        self,
        num_edges: Sequence[int],
    ) -> list[Vec3D]:
        """Generate `self.edges` if any, and then additionally generate random
        edges for each prespecified bound.
        """
        assert len(num_edges) == len(self.bounds)
        assert all(n_edge >= 0 for n_edge in num_edges)

        result = [Vec3D(*edge) for edge in self.edges]

        # Sample random edges for each bound
        for bound, n_edge in zip(self.bounds, num_edges):
            sampled = 0
            while sampled < n_edge:
                edge = Vec3D(*[random.randint(-abs(bnd), abs(bnd)) for bnd in bound])
                if edge == Vec3D(0, 0, 0):
                    continue
                result.append(edge)
                sampled += 1

        return result


@typechecked
def compute_affinity(
    data1: torch.Tensor,
    data2: torch.Tensor,
    dim: int = -4,
    keepdims: bool = True,
) -> torch.Tensor:
    """Compute an affinity map from a pair of embeddings based on l2 distance."""
    dist = torch.sum((data1 - data2) ** 2, dim=dim, keepdim=keepdims)
    result = torch.exp(-dist)
    return result


@typechecked
class EdgeDecoder:
    def __call__(self, data: torch.Tensor, disp: Vec3D) -> torch.Tensor:
        pair = tensor_ops.get_disp_pair(data, disp)
        result = compute_affinity(pair[0], pair[1])
        return result


@builder.register("EdgeLoss")
@typechecked
class EdgeLoss(nn.Module):
    def __init__(
        self,
        bounds: Sequence[Sequence[int]],
        num_edges: Sequence[int],
        edges: Sequence[Sequence[int]] | None = None,
        reduction: Literal["mean", "sum", "none"] = "none",
        balancer: nn.Module | None = None,
    ) -> None:
        super().__init__()
        assert len(num_edges) == len(bounds)
        self.num_edges = list(num_edges)
        self.sampler = EdgeSampler(edges, bounds)
        self.decoder = EdgeDecoder()
        self.criterion = EdgeCRF(
            partial(
                LossWithMask,
                criterion=nn.BCEWithLogitsLoss,
                reduction="none",
            ),
            reduction,
            balancer,
        )

    def forward(
        self,
        pred: torch.Tensor,
        trgt: torch.Tensor,
        mask: torch.Tensor,
        splt: torch.Tensor | None = None,
    ) -> torch.Tensor | list[torch.Tensor] | None:
        preds = []  # type: list[torch.Tensor]
        trgts = []
        masks = []
        edges = self.sampler.generate_edges(self.num_edges)
        for edge in edges:
            aff, msk = self._generate_target(trgt, mask, edge, splt)
            preds.append(self.decoder(pred, edge))
            trgts.append(aff)
            masks.append(msk)
        return self.criterion(preds, trgts, masks)

    @staticmethod
    def _generate_target(
        trgt: torch.Tensor,
        mask: torch.Tensor,
        edge: Vec3D,
        splt: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Ignore background
        mask *= (trgt != 0).to(mask.dtype)
        aff, msk = tensor_ops.seg_to_aff(trgt, edge, mask=mask)

        # Mask out interactions between local split by connected components
        if splt is not None:
            splt_aff = tensor_ops.seg_to_aff(splt, edge)
            msk[aff != splt_aff] = 0

        return aff, msk
