from __future__ import annotations

import random
from typing import Literal, Sequence

import attrs
import torch
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.layer import JointIndexDataProcessor
from zetta_utils.layer.volumetric import VolumetricIndex

from .common import ComposedAugment


@builder.register("FlipAugment")
@typechecked
@attrs.frozen
class FlipAugment(JointIndexDataProcessor):
    dims: Sequence[int]
    prob: float = 0.5

    def process_index(
        self, idx: VolumetricIndex, mode: Literal["read", "write"]
    ) -> VolumetricIndex:
        return idx

    def process_data(
        self, data: dict[str, torch.Tensor], mode: Literal["read", "write"]
    ) -> dict[str, torch.Tensor]:
        coin = random.uniform(0, 1)
        if coin < self.prob:
            for key, val in data.items():
                data[key] = torch.flip(val, tuple(self.dims))
        return data


@builder.register("TransposeAugment")
@typechecked
@attrs.mutable
class TransposeAugment(JointIndexDataProcessor):
    dim0: int
    dim1: int
    local: bool = True
    prob: float = 0.5

    prepared_coin: bool | None = attrs.field(init=False, default=None)

    def process_index(
        self, idx: VolumetricIndex, mode: Literal["read", "write"]
    ) -> VolumetricIndex:
        coin = random.uniform(0, 1)  # Coin flip
        self.prepared_coin = coin < self.prob
        if self.prepared_coin:
            idx = idx.transposed(self.dim0, self.dim1, self.local)
        return idx

    def process_data(
        self, data: dict[str, torch.Tensor], mode: Literal["read", "write"]
    ) -> dict[str, torch.Tensor]:
        assert self.prepared_coin is not None

        if self.prepared_coin:
            for key, val in data.items():
                data[key] = torch.transpose(val, self.dim0, self.dim1)

        self.prepared_coin = None
        return data


@builder.register("SimpleAugment")
@typechecked
def build_simple_augment(isotropic: bool = False) -> JointIndexDataProcessor:
    augments = [
        FlipAugment(dims=(-3,)),
        FlipAugment(dims=(-2,)),
        FlipAugment(dims=(-1,)),
        TransposeAugment(dim0=-3, dim1=-2),  # xy
    ]

    if isotropic:
        augments += [
            TransposeAugment(dim0=-2, dim1=-1),  # yz
            TransposeAugment(dim0=-1, dim1=-3),  # zx
        ]

    result = ComposedAugment(augments)
    return result
