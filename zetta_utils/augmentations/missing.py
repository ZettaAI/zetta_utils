from __future__ import annotations

import attrs
import torch
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.distributions import Distribution, to_distribution

from .section import QuadrantWiseAugmentMixin, SectionWiseAugment


@builder.register("MissingSectionAugment")
@typechecked
@attrs.mutable
class MissingSectionAugment(SectionWiseAugment):
    fill: float | Distribution = 0
    per_section: bool = False

    fill_distr: Distribution = attrs.field(init=False)

    prepared_fill: float | None = attrs.field(init=False, default=None)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.fill_distr = to_distribution(self.fill)

    def prepare(self) -> None:
        super().prepare()
        self.prepared_fill = self.fill_distr()

    def process_section(self, data: torch.Tensor) -> torch.Tensor:
        if self.per_section:
            value = self.fill_distr()
        else:
            assert self.prepared_fill is not None
            value = self.prepared_fill
        result = data.fill_(value)
        return result


@builder.register("PartialMissingSectionAugment")
@typechecked
@attrs.mutable
class PartialMissingSectionAugment(MissingSectionAugment, QuadrantWiseAugmentMixin):
    per_quadrant: bool = False

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        QuadrantWiseAugmentMixin.init(self)

    def process_section(self, data: torch.Tensor) -> torch.Tensor:
        if self.per_section:
            self.prepared_fill = self.fill_distr()
        result = self.process_section_quadrant_wise(data)
        return result

    def process_quadrant(self, data: torch.Tensor) -> torch.Tensor:
        if self.per_quadrant:
            value = self.fill_distr()
        else:
            assert self.prepared_fill is not None
            value = self.prepared_fill
        result = data.fill_(value)
        return result
