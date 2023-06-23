from __future__ import annotations

from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.distributions import Distribution

from .section import RandomSection
from .transform import PartialSection, RandomFill


@builder.register("MissingSection")
@typechecked
def build_missing_section(
    prob: float,
    key: str,
    rate: float | Distribution,
    fill: float | Distribution = 0,
    per_section: float = False,
) -> RandomSection:
    return RandomSection(
        prob=prob,
        key=key,
        rate=rate,
        transform=RandomFill(fill=fill, frozen=not per_section),
    )


@builder.register("PartialMissingSection")
@typechecked
def build_partial_missing_section(
    prob: float,
    key: str,
    rate: float | Distribution,
    fill: float | Distribution = 0,
    per_section: bool = False,
    per_partial: bool = False,
) -> RandomSection:
    return RandomSection(
        prob=prob,
        key=key,
        rate=rate,
        transform=PartialSection(
            transform=RandomFill(
                fill=fill,
                frozen=not per_partial,
            ),
            frozen=not per_section,
            per_partial=per_partial,
        ),
    )
