from __future__ import annotations
from typing import Union, Protocol, runtime_checkable, overload, Optional
import numpy as np
import numpy.typing as npt

from typeguard import typechecked
from zetta_utils import builder
from zetta_utils.partial import ComparablePartial
from zetta_utils.typing import Number

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
    def __call__(self, size: Optional[NoneType] = None) -> Number:
        ...


@builder.register("uniform_dist")
@typechecked
def uniform_dist(low: Number = 0.0, high: Number = 1.0) -> Distribution:  # pragma: no cover
    # We know that this comparable becomes a distribution, so can ignore types here
    return ComparablePartial(np.random.uniform, low=low, high=high)  # type: ignore


@builder.register("normal_dist")
@typechecked
def normal_dist_dist(loc: Number = 0.0, scale: Number = 1.0) -> Distribution:  # pragma: no cover
    # We know that this comparable becomes a distribution, so can ignore types here
    return ComparablePartial(np.random.normal, loc=loc, scale=scale)  # type: ignore


@typechecked
def to_distribution(x: Union[Distribution, Number]) -> Distribution:
    result: Distribution
    if isinstance(x, (int, float)):
        result = uniform_dist(x, x)
    else:
        result = x
    return result
