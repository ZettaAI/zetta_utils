from __future__ import annotations

from typing import Optional, Protocol, Union, overload, runtime_checkable

import numpy as np
import numpy.typing as npt
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.common.partial import ComparablePartial

NoneType = type(None)


@runtime_checkable
class Distribution(Protocol):  # pragma: no cover
    @overload
    def __call__(self, size: int) -> npt.NDArray:
        ...

    @overload
    def __call__(self, size: tuple[int]) -> npt.NDArray:
        ...

    @overload
    def __call__(self, size: None = None) -> float:
        ...


@builder.register("uniform_distr")
@typechecked
def uniform_distr(low: float = 0.0, high: float = 1.0) -> Distribution:  # pragma: no cover
    # We know that this comparable becomes a distribution, so can ignore types here
    return ComparablePartial(np.random.uniform, low=low, high=high)  # type: ignore


@builder.register("normal_distr")
@typechecked
def normal_distr(loc: float = 0.0, scale: float = 1.0) -> Distribution:  # pragma: no cover
    # We know that this comparable becomes a distribution, so can ignore types here
    return ComparablePartial(np.random.normal, loc=loc, scale=scale)  # type: ignore


@typechecked
def to_distribution(x: Union[Distribution, float]) -> Distribution:
    result: Distribution
    if isinstance(x, (int, float)):
        result = uniform_distr(x, x)
    else:
        result = x
    return result
