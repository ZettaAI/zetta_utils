from __future__ import annotations

from functools import partial
from typing import Callable, Literal

import attrs
import skimage
import torch
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.distributions import Distribution, to_distribution
from zetta_utils.layer import JointIndexDataProcessor
from zetta_utils.layer.volumetric import VolumetricIndex
from zetta_utils.tensor_ops import convert


@builder.register("AdditiveGaussianNoiseAugment")
@typechecked
@attrs.mutable
class AdditiveGaussianNoiseAugment(JointIndexDataProcessor):
    key: str
    scale: float | Distribution
    mean: float = 0

    scale_distr: Distribution = attrs.field(init=False)

    prepared_aug: Callable | None = attrs.field(init=False, default=None)

    def __attrs_post_init__(self):
        self.scale_distr = to_distribution(self.scale)

    def process_index(
        self, idx: VolumetricIndex, mode: Literal["read", "write"]
    ) -> VolumetricIndex:
        return idx

    def process_data(
        self, data: dict[str, torch.Tensor], mode: Literal["read", "write"]
    ) -> dict[str, torch.Tensor]:
        self.prepare()
        assert self.prepared_aug is not None
        assert self.key in data
        raw = data[self.key]
        data_np = convert.to_np(raw)
        assert data_np.min() >= -1 and data_np.max() <= 1
        data[self.key] = convert.astype(self.prepared_aug(data_np), raw, cast=True)
        return data

    def prepare(self) -> None:
        self.prepared_aug = partial(
            skimage.util.random_noise,
            mode="gaussian",
            clip=True,
            mean=self.mean,
            var=self.scale_distr() ** 2,
        )
