from __future__ import annotations

import attrs
import torch
from scipy.ndimage import gaussian_filter
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.distributions import Distribution, to_distribution
from zetta_utils.tensor_ops import convert
from zetta_utils.tensor_typing import TensorTypeVar

from .section import QuadrantWiseAugmentMixin, SectionWiseAugment


@typechecked
def apply_gaussian_filter(
    data: TensorTypeVar, sigma: float | tuple[float, float, float]
) -> TensorTypeVar:
    data_np = convert.to_np(data)
    blurred = gaussian_filter(data_np, sigma=sigma)
    result = convert.astype(blurred, data, cast=True)
    return result


@builder.register("BlurrySectionAugment")
@typechecked
@attrs.mutable
class BlurrySectionAugment(SectionWiseAugment):
    sigma: float | Distribution
    per_section: bool = False

    sigma_distr: Distribution = attrs.field(init=False)

    prepared_sigma: float | None = attrs.field(init=False, default=None)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.sigma_distr = to_distribution(self.sigma)

    def prepare(self) -> None:
        super().prepare()
        self.prepared_sigma = self.sigma_distr()

    def process_section(self, data: torch.Tensor) -> torch.Tensor:
        if self.per_section:
            sigma = self.sigma_distr()
        else:
            assert self.prepared_sigma is not None
            sigma = self.prepared_sigma
        result = apply_gaussian_filter(data, sigma)
        return result


@builder.register("PartialBlurrySectionAugment")
@typechecked
@attrs.mutable
class PartialBlurrySectionAugment(QuadrantWiseAugmentMixin, BlurrySectionAugment):
    per_quadrant: bool = False

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        QuadrantWiseAugmentMixin.init(self)

    def process_section(self, data: torch.Tensor) -> torch.Tensor:
        if self.per_section:
            self.prepared_sigma = self.sigma_distr()
        result = self.process_section_quadrant_wise(data)
        return result

    def process_quadrant(self, data: torch.Tensor) -> torch.Tensor:
        if self.per_quadrant:
            sigma = self.sigma_distr()
        else:
            assert self.prepared_sigma is not None
            sigma = self.prepared_sigma
        result = apply_gaussian_filter(data, sigma)
        return result
