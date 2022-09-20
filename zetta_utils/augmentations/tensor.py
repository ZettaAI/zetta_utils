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
    data_torch = tensor_ops.to_torch(data)
    if mask_fn is not None:
        mask = tensor_ops.to_torch(mask_fn(data_torch))
    else:
        mask = torch.ones_like(data_torch, dtype=torch.bool)

    result = tensor_ops.astype(mask, data)
    return result


@builder.register("brightness_aug")
@typechecked
@prob_aug
def add_scalar_aug(
    data: TensorTypeVar,
    value_distr: Union[distributions.Distribution, Number],
    mask_fn: Optional[Callable[..., Tensor]] = None,
) -> TensorTypeVar:
    data_torch = tensor_ops.to_torch(data).float()
    weights_mask = _get_weights_mask(data_torch, mask_fn).float()
    value = distributions.to_distribution(value_distr)() * weights_mask
    data_torch += value
    result = tensor_ops.astype(data_torch, data)
    return result


@builder.register("clamp_values_aug")
@typechecked
@prob_aug
def clamp_values_aug(
    data: TensorTypeVar,
    low_distr: Optional[Union[distributions.Distribution, Number]] = None,
    high_distr: Optional[Union[distributions.Distribution, Number]] = None,
    mask_fn: Optional[Callable[..., Tensor]] = None,
):
    data_torch = tensor_ops.to_torch(data).float()
    mask = _get_weights_mask(data_torch, mask_fn).bool()

    if high_distr is not None:
        high = distributions.to_distribution(high_distr)()
        data_torch[mask & (data_torch > high)] = high

    if low_distr is not None:
        low = distributions.to_distribution(low_distr)()
        data_torch[mask & (data_torch < low)] = low

    result = tensor_ops.astype(data_torch, data)
    return result
