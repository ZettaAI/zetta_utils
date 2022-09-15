from typing import Union, Optional, Callable
import torch
from typeguard import typechecked

from zetta_utils import distributions, tensor_ops, builder
from zetta_utils.typing import TensorTypeVar, Number, Tensor

from .common import prob_aug


@builder.register("brightness_aug")
@typechecked
@prob_aug
def brightness_aug(
    data: TensorTypeVar,
    adj: Union[distributions.Distribution, Number],
    low_cap: Optional[Union[distributions.Distribution, Number]] = None,
    high_cap: Optional[Union[distributions.Distribution, Number]] = None,
    mask_fn: Optional[Callable[..., Tensor]] = None,
) -> TensorTypeVar:
    data_ = tensor_ops.to_torch(data).float()
    if mask_fn is not None:
        mask = tensor_ops.to_torch(mask_fn(data_))
    else:
        mask = torch.ones_like(data_, dtype=torch.bool)

    adj_v = distributions.to_distribution(adj)()
    data_[mask] += adj_v

    if high_cap is not None:
        high_cap_v = distributions.to_distribution(high_cap)()
        data_[mask & (data_ > high_cap_v)] = high_cap_v

    if low_cap is not None:
        low_cap_v = distributions.to_distribution(low_cap)()
        data_[mask & (data_ < low_cap_v)] = low_cap_v

    result = tensor_ops.astype(data_, data)
    return result
