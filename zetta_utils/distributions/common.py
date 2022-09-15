from typing import Union, Protocol, runtime_checkable
import random

from typeguard import typechecked
from zetta_utils import builder
from zetta_utils.partial import ComparablePartial
from zetta_utils.typing import Number


@runtime_checkable
class Distribution(Protocol):  # pragma: no cover
    def __call__(self, /) -> Number:
        ...


@builder.register("uniform_dist")
@typechecked
def uniform_dist(a: Number, b: Number) -> Distribution:  # pragma: no cover
    return ComparablePartial(random.uniform, a=a, b=b)


@builder.register("gauss_dist")
@typechecked
def gauss_dist(mu: Number, sigma: Number) -> Distribution:  # pragma: no cover
    return ComparablePartial(random.gauss, mu=mu, sigma=sigma)


@typechecked
def to_distribution(x: Union[Distribution, Number]) -> Distribution:
    result: Distribution
    if isinstance(x, (int, float)):
        result = uniform_dist(x, x)
    else:
        result = x
    return result
