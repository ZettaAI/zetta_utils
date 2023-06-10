from __future__ import annotations

import math
from abc import abstractmethod
from typing import Literal

import attrs
import numpy as np
import torch
from typeguard import typechecked

from zetta_utils.distributions import Distribution, to_distribution, uniform_distr
from zetta_utils.layer import JointIndexDataProcessor
from zetta_utils.layer.volumetric import VolumetricIndex


@typechecked
@attrs.mutable
class SectionWiseAugment(JointIndexDataProcessor):
    key: str
    prob: float | Distribution

    distr: Distribution = attrs.field(init=False)

    prepared_prob: float | None = attrs.field(init=False, default=None)

    def __attrs_post_init__(self):
        distr = to_distribution(self.prob)
        object.__setattr__(self, "distr", distr)

    def process_index(
        self, idx: VolumetricIndex, mode: Literal["read", "write"]
    ) -> VolumetricIndex:
        return idx

    def process_data(
        self, data: dict[str, torch.Tensor], mode: Literal["read", "write"]
    ) -> dict[str, torch.Tensor]:
        assert self.key in data
        raw = data[self.key]

        # Prepare for processing
        self.prepare()

        # Process each section
        num_z = raw.shape[-1]
        coins = np.random.uniform(0, 1, num_z)
        for z in range(num_z):
            assert self.prepared_prob is not None
            if coins[z] < self.prepared_prob:
                raw[..., z] = self.process_section(raw[..., z])

        data[self.key] = raw
        return data

    def prepare(self) -> None:
        self.prepared_prob = self.distr()

    @abstractmethod
    def process_section(self, data: torch.Tensor) -> torch.Tensor:
        ...


@typechecked
class QuadrantWiseAugmentMixin:
    def init(self) -> None:
        self.distr_x: Distribution = uniform_distr()
        self.distr_y: Distribution = uniform_distr()

    def process_section_quadrant_wise(self, data: torch.Tensor) -> torch.Tensor:
        # Pick a random pivot point
        ratio_x = self.distr_x()
        ratio_y = self.distr_y()
        pivot_x = int(math.floor(ratio_x * data.shape[-2]))
        pivot_y = int(math.floor(ratio_y * data.shape[-1]))

        # Process each quadrant independently
        coins = np.random.uniform(0, 1, 4)
        quads = [
            (slice(None, pivot_x), slice(None, pivot_y)),
            (slice(pivot_x, None), slice(None, pivot_y)),
            (slice(None, pivot_x), slice(pivot_y, None)),
            (slice(pivot_x, None), slice(pivot_y, None)),
        ]
        for coin, quad in zip(coins, quads):
            if coin < 0.5:
                data[..., quad[0], quad[1]] = self.process_quadrant(data[..., quad[0], quad[1]])

        return data

    @abstractmethod
    def process_quadrant(self, data: torch.Tensor) -> torch.Tensor:
        ...
