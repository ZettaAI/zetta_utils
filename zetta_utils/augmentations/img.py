from typing import Union, Optional, Callable
import torch
from typeguard import typechecked

from zetta_utils import distributions, tensor_ops, builder
from zetta_utils.typing import TensorTypeVar, Number, Tensor

from .common import prob_aug


def _get_weights_mask(
    data: TensorTypeVar,
    mask_fn: Optional[Callable[..., Tensor]] = None,
) -> TensorTypeVar:
    data_ = tensor_ops.to_torch(data)
    if mask_fn is not None:
        mask = tensor_ops.to_torch(mask_fn(data_))
    else:
        mask = torch.ones_like(data_, dtype=torch.bool)

    result = tensor_ops.astype(mask, data)
    return result


@builder.register("brightness_aug")
@typechecked
@prob_aug
def brightness_aug(
    data: TensorTypeVar,
    adj: Union[distributions.Distribution, Number],
    mask_fn: Optional[Callable[..., Tensor]] = None,
) -> TensorTypeVar:
    data_ = tensor_ops.to_torch(data).float()
    weights_mask = _get_weights_mask(data_, mask_fn).float()
    adj_v = distributions.to_distribution(adj)() * weights_mask
    data_ += adj_v
    result = tensor_ops.astype(data_, data)
    return result


@builder.register("clamp_values_aug")
@typechecked
@prob_aug
def clamp_values_aug(
    data: TensorTypeVar,
    low: Optional[Union[distributions.Distribution, Number]] = None,
    high: Optional[Union[distributions.Distribution, Number]] = None,
    mask_fn: Optional[Callable[..., Tensor]] = None,
):
    data_ = tensor_ops.to_torch(data).float()
    mask = _get_weights_mask(data_, mask_fn).bool()

    if high is not None:
        high_v = distributions.to_distribution(high)()
        data_[mask & (data_ > high_v)] = high_v

    if low is not None:
        low_v = distributions.to_distribution(low)()
        data_[mask & (data_ < low_v)] = low_v

    result = tensor_ops.astype(data_, data)
    return result
