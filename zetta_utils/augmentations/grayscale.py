from __future__ import annotations

from functools import partial
from typing import Callable

import attrs
import torch
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.distributions import Distribution, to_distribution

from .common import DataAugment
from .section import RandomSection
from .transform import DataTransform, PartialSection


@typechecked
def grayscale_jitter(
    data: torch.Tensor,
    contrast: float,
    brightness: float,
    gamma: float,
    gamma_base: float = 2.0,
) -> torch.Tensor:
    data *= contrast
    data += brightness
    data = torch.clamp(data, min=0, max=1)
    data **= gamma_base ** gamma
    return data


@typechecked
@attrs.mutable
class GrayscaleJitter(DataTransform):
    contrast: float | Distribution = 1.0
    brightness: float | Distribution = 0.0
    gamma: float | Distribution = 0.0

    frozen: bool = False

    contrast_distr: Distribution = attrs.field(init=False)
    brightness_distr: Distribution = attrs.field(init=False)
    gamma_distr: Distribution = attrs.field(init=False)

    prepared_jitter: Callable | None = attrs.field(init=False, default=None)

    def __attrs_post_init__(self):
        self.contrast_distr = to_distribution(self.contrast)
        self.brightness_distr = to_distribution(self.brightness)
        self.gamma_distr = to_distribution(self.gamma)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if self.frozen:
            assert self.prepared_jitter is not None
            jitter = self.prepared_jitter
        else:
            jitter = self.random_jitter()
        return jitter(data)

    def prepare(self) -> None:
        self.prepared_jitter = self.random_jitter()

    def random_jitter(self) -> Callable:
        return partial(
            grayscale_jitter,
            contrast=self.contrast_distr(),
            brightness=self.brightness_distr(),
            gamma=self.gamma_distr(),
        )


@builder.register("GrayscaleJitter3D")
@typechecked
@attrs.frozen
class GrayscaleJitter3D(DataAugment):
    key: str

    contrast: float | Distribution = 1.0
    brightness: float | Distribution = 0.0
    gamma: float | Distribution = 0.0

    random_jitter: GrayscaleJitter = attrs.field(init=False)

    def __attrs_post_init__(self) -> None:
        random_jitter = GrayscaleJitter(
            contrast=self.contrast,
            brightness=self.brightness,
            gamma=self.gamma,
            frozen=False,
        )
        object.__setattr__(self, "random_jitter", random_jitter)

    def augment(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        assert self.key in data
        raw = data[self.key]
        data[self.key] = self.random_jitter(raw)
        return data


@builder.register("GrayscaleJitter2D")
@typechecked
def build_grayscale_jitter_2d(
    prob: float,
    key: str,
    rate: float | Distribution = 1.0,
    contrast: float | Distribution = 1.0,
    brightness: float | Distribution = 0.0,
    gamma: float | Distribution = 0.0,
    per_section: float = False,
) -> RandomSection:
    return RandomSection(
        prob=prob,
        key=key,
        rate=rate,
        transform=GrayscaleJitter(
            contrast=contrast,
            brightness=brightness,
            gamma=gamma,
            frozen=not per_section,
        ),
    )


@builder.register("PartialGrayscaleJitter2D")
@typechecked
def build_partial_grayscale_jitter_2d(
    prob: float,
    key: str,
    rate: float | Distribution = 1.0,
    contrast: float | Distribution = 1.0,
    brightness: float | Distribution = 0.0,
    gamma: float | Distribution = 0.0,
    per_section: bool = False,
    per_partial: bool = False,
) -> RandomSection:
    return RandomSection(
        prob=prob,
        key=key,
        rate=rate,
        transform=PartialSection(
            transform=GrayscaleJitter(
                contrast=contrast,
                brightness=brightness,
                gamma=gamma,
                frozen=not per_partial,
            ),
            frozen=not per_section,
            per_partial=per_partial,
        ),
    )
