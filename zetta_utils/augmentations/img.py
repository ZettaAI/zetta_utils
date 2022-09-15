from typing import Union, Optional, overload, Callable, TypeVar
import torch
import numpy.typing as npt
from typeguard import typechecked

from zetta_utils import distributions, tensor_ops
from zetta_utils.typing import TensorTypeVar, Number, Tensor

from .common import prob_aug

'''
def brightness_aug(
    data: npt.NDArray,
    adj: Union[distributions.Distribution, Number],
    low_cap: Optional[Union[distributions.Distribution, Number]] = ...,
    high_cap: Optional[Union[distributions.Distribution, Number]] = ...,
    mask_fn: Optional[Callable[..., Tensor]] = ...,
) -> npt.NDArray:
    ...

@overload
def brightness_aug(
    data: torch.Tensor,
    adj: Union[distributions.Distribution, Number],
    low_cap: Optional[Union[distributions.Distribution, Number]] = ...,
    high_cap: Optional[Union[distributions.Distribution, Number]] = ...,
    mask_fn: Optional[Callable[..., Tensor]] = ...,
) -> torch.Tensor:
    ...
'''

@typechecked
@prob_aug
def brightness_aug(
    data: TensorTypeVar,
    adj: Union[distributions.Distribution, Number],
    low_cap: Optional[Union[distributions.Distribution, Number]] = None,
    high_cap: Optional[Union[distributions.Distribution, Number]] = None,
    mask_fn: Optional[Callable[..., Tensor]] = None,
) -> TensorTypeVar:
    result = tensor_ops.to_torch(data)
    breakpoint()
    if mask_fn is not None:
        mask = tensor_ops.to_torch(mask_fn(data))
    else:
        mask = torch.ones_like(result)

    adj_v = distributions.to_distribution(adj).rvs()
    result[mask] += adj_v

    if high_cap is not None:
        high_cap_v = distributions.to_distribution(high_cap).rvs()
        result[mask & (result > high_cap_v)] = high_cap_v

    if low_cap is not None:
        low_cap_v = distributions.to_distribution(low_cap).rvs()
        result[mask & (result < low_cap_v)] = low_cap_v

    result = tensor_ops.astype(result, data)
    return result
