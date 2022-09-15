from typing import Union, Protocol, runtime_checkable, overload, Literal
import scipy # type: ignore

from typeguard import typechecked
import numpy.typing as npt
from zetta_utils import builder

@runtime_checkable
class Distribution(Protocol):
    def rvs(self) -> Union[float, int]:
        ...

    @overload
    def rvs(self, *args, size: Literal[1], **kwargs) -> Union[float, int]:
        ...

    def rvs(self, *args, size: int = ..., **kwargs) -> Union[npt.NDArray, float, int]:
        ...


#Distribution = Union[
#    scipy.stats._distn_infrastructure.rv_generic, # pylint: disable=protected-access
#    scipy.stats._distn_infrastructure.rv_frozen, # pylint: disable=protected-access
#]

@typechecked
def to_distribution(
        x: Union[Distribution, int, float]
) -> Distribution:
    result: Distribution
    if isinstance(x, (int, float)):
        result = scipy.stats._continuous_distns.uniform_gen(a=x, b=x)
    else:
        result = x
    return result

builder.register("uniform_dist")(scipy.stats.uniform)
builder.register("norm_dist")(scipy.stats.norm)
