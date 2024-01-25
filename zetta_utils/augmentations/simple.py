from __future__ import annotations

from functools import partial
from typing import Sequence

import attrs
import torch
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.layer import JointIndexDataProcessor
from zetta_utils.layer.volumetric import VolumetricIndex

from .common import ComposedAugment, DataAugment, JointIndexDataAugment


@builder.register("RandomFlip")
@typechecked
@attrs.frozen
class RandomFlip(DataAugment):  # pragma: no cover
    dims: Sequence[int]

    def augment(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for key, val in data.items():
            data[key] = torch.flip(val, tuple(self.dims))
        return data


@builder.register("RandomTranspose")
@typechecked
@attrs.mutable
class RandomTranspose(JointIndexDataAugment):  # pragma: no cover
    dim0: int
    dim1: int

    local: bool = True

    def augment_index(self, idx: VolumetricIndex) -> VolumetricIndex:
        return idx.transposed(self.dim0, self.dim1, self.local)

    def augment_data(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for key, val in data.items():
            data[key] = torch.transpose(val, self.dim0, self.dim1)
        return data


@builder.register("SimpleAugment")
@typechecked
def build_simple_augment(prob: float = 0.5, isotropic: bool = False) -> JointIndexDataProcessor:
    """Flip & rotate by 90 degree."""

    # Pre-configure RandomFlip and RandomTranspose with prob
    flip = partial(RandomFlip, prob=prob)
    transpose = partial(RandomTranspose, prob=prob)

    # Anisotropic simple augment
    augments: list[DataAugment | JointIndexDataAugment] = [
        flip(dims=(-3,)),
        flip(dims=(-2,)),
        flip(dims=(-1,)),
        transpose(dim0=-3, dim1=-2),  # xy
    ]

    # Isotropic simple augment
    if isotropic:
        augments += [
            transpose(dim0=-2, dim1=-1),  # yz
            transpose(dim0=-1, dim1=-3),  # zx
        ]

    return ComposedAugment(augments)
