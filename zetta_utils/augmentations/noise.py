from __future__ import annotations

from functools import partial
from typing import Callable

import attrs
import skimage
import torch
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.distributions import Distribution, to_distribution
from zetta_utils.tensor_ops import convert

from .common import DataAugment


@builder.register("RandomGaussianNoise")
@typechecked
@attrs.frozen
class RandomGaussianNoise(DataAugment):  # pragma: no cover
    key: str
    std: float | Distribution

    mean: float = 0

    std_distr: Distribution = attrs.field(init=False)
    random_noise: Callable = attrs.field(init=False)

    def __attrs_post_init__(self):
        object.__setattr__(self, "std_distr", to_distribution(self.std))
        object.__setattr__(
            self,
            "random_noise",
            partial(
                skimage.util.random_noise,
                mode="gaussian",
                clip=True,
                mean=self.mean,
            ),
        )

    def augment(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        assert self.key in data
        raw = data[self.key]
        data_np = convert.to_np(raw)
        assert data_np.min() >= 0 and data_np.max() <= 1
        result = self.random_noise(data_np, var=self.std_distr() ** 2)
        data[self.key] = convert.astype(result, raw, cast=True)
        return data
