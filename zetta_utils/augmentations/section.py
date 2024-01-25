from __future__ import annotations

import attrs
import numpy as np
import torch
from typeguard import typechecked

from zetta_utils.distributions import Distribution, to_distribution

from .common import DataAugment
from .transform import DataTransform


@typechecked
@attrs.mutable
class RandomSection(DataAugment):
    key: str
    rate: float | Distribution
    transform: DataTransform

    rate_distr: Distribution = attrs.field(init=False)

    prepared_rate: float | None = attrs.field(init=False, default=None)

    def __attrs_post_init__(self):
        self.rate_distr = to_distribution(self.rate)

    def augment(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        assert self.key in data
        raw = data[self.key]

        # Prepare for processing
        self.prepare()

        # Process random sections
        num_z = raw.shape[-1]
        coins = np.random.uniform(0, 1, num_z)
        for z in range(num_z):
            assert self.prepared_rate is not None
            if coins[z] < self.prepared_rate:
                raw[..., z] = self.transform(raw[..., z])

        data[self.key] = raw
        return data

    def prepare(self) -> None:
        self.prepared_rate = self.rate_distr()
        self.transform.prepare()
