from __future__ import annotations

from functools import partial
from typing import Callable, Literal

import attrs
import torch
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.distributions import Distribution, to_distribution
from zetta_utils.layer import JointIndexDataProcessor
from zetta_utils.layer.volumetric import VolumetricIndex

from .section import QuadrantWiseAugmentMixin, SectionWiseAugment


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


@builder.register("GrayscaleJitter")
@typechecked
@attrs.frozen
class GrayscaleJitter:
    contrast: float | Distribution = 1.0
    brightness: float | Distribution = 0.0
    gamma: float | Distribution = 0.0

    contrast_distr: Distribution = attrs.field(init=False)
    brightness_distr: Distribution = attrs.field(init=False)
    gamma_distr: Distribution = attrs.field(init=False)

    def __attrs_post_init__(self):
        object.__setattr__(self, "contrast_distr", to_distribution(self.contrast))
        object.__setattr__(self, "brightness_distr", to_distribution(self.brightness))
        object.__setattr__(self, "gamma_distr", to_distribution(self.gamma))

    def __call__(self) -> Callable:
        contrast = self.contrast_distr()
        brightness = self.brightness_distr()
        gamma = self.gamma_distr()
        return partial(grayscale_jitter, contrast=contrast, brightness=brightness, gamma=gamma)


@builder.register("GrayscaleJitter3D")
@typechecked
@attrs.frozen
class GrayscaleJitter3D(JointIndexDataProcessor):
    key: str
    jitter: GrayscaleJitter

    def process_index(
        self, idx: VolumetricIndex, mode: Literal["read", "write"]
    ) -> VolumetricIndex:
        return idx

    def process_data(
        self, data: dict[str, torch.Tensor], mode: Literal["read", "write"]
    ) -> dict[str, torch.Tensor]:
        assert self.key in data
        raw = data[self.key]
        result = self.jitter()(raw)
        data[self.key] = result
        return data


@builder.register("GrayscaleJitter2D")
@typechecked
@attrs.mutable
class GrayscaleJitter2D(SectionWiseAugment):
    jitter: GrayscaleJitter

    per_section: bool = False

    prepared_jitter: Callable | None = attrs.field(init=False, default=None)

    def prepare(self) -> None:
        super().prepare()
        self.prepared_jitter = self.jitter()

    def process_section(self, data: torch.Tensor) -> torch.Tensor:
        if self.per_section:
            jitter = self.jitter()
        else:
            assert self.prepared_jitter is not None
            jitter = self.prepared_jitter
        result = jitter(data)
        return result


@builder.register("PartialGrayscaleJitter2D")
@typechecked
@attrs.mutable
class PartialGrayscaleJitter2D(GrayscaleJitter2D, QuadrantWiseAugmentMixin):
    per_quadrant: bool = False

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        QuadrantWiseAugmentMixin.init(self)

    def process_section(self, data: torch.Tensor) -> torch.Tensor:
        if self.per_section:
            self.prepared_jitter = self.jitter()
        result = self.process_section_quadrant_wise(data)
        return result

    def process_quadrant(self, data: torch.Tensor) -> torch.Tensor:
        if self.per_quadrant:
            jitter = self.jitter()
        else:
            assert self.prepared_jitter is not None
            jitter = self.prepared_jitter
        result = jitter(data)
        return result
