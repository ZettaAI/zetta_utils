from __future__ import annotations

from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.distributions import Distribution

from .section import RandomSection
from .transform import GaussianBlur, PartialSection


@builder.register("BlurrySection")
@typechecked
def build_blurry_section(
    prob: float,
    key: str,
    rate: float | Distribution,
    sigma: float | Distribution,
    per_section: float = False,
) -> RandomSection:
    return RandomSection(
        prob=prob,
        key=key,
        rate=rate,
        transform=GaussianBlur(sigma=sigma, frozen=not per_section),
    )


@builder.register("PartialBlurrySection")
@typechecked
def build_partial_blurry_section(
    prob: float,
    key: str,
    rate: float | Distribution,
    sigma: float | Distribution,
    per_section: bool = False,
    per_partial: bool = False,
) -> RandomSection:
    return RandomSection(
        prob=prob,
        key=key,
        rate=rate,
        transform=PartialSection(
            transform=GaussianBlur(
                sigma=sigma,
                frozen=not per_partial,
            ),
            frozen=not per_section,
            per_partial=per_partial,
        ),
    )
