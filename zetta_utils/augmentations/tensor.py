from typing import Union, Optional, Callable
import math
import random
import torch
from torchvision.transforms.functional import rotate  # type: ignore
from typeguard import typechecked

from zetta_utils import distributions, tensor_ops, builder
from zetta_utils.typing import Number
from zetta_utils.tensor_typing import TensorTypeVar, Tensor

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


def _random_square_tile_pattern(
    data: torch.Tensor,
    *,
    tile_stride: Union[distributions.Distribution, Number],
    tile_size: Union[distributions.Distribution, Number],
    rotation_degree: Union[distributions.Distribution, Number],
) -> torch.Tensor:
    tile_stride = int(distributions.to_distribution(tile_stride)())
    tile_size = int(distributions.to_distribution(tile_size)())
    rotation_degree = distributions.to_distribution(rotation_degree)()

    if rotation_degree != 0.0:
        # TODO: Could reduce padding based on actual angle - right now assuming worst-case 45°
        rotation_padding = 0.5 * (math.hypot(data.shape[-1], data.shape[-1]) - data.shape[-1])
        rotation_padding = tile_stride * math.ceil(rotation_padding / tile_stride)
    else:
        rotation_padding = 0

    # Tile pattern
    full_padding = rotation_padding + tile_stride
    tile_count = math.ceil((data.shape[-1] + 2 * full_padding) / tile_stride)
    pattern = torch.conv_transpose2d(
        torch.ones(1, 1, tile_count, tile_count, device=data.device),
        weight=torch.ones(1, 1, tile_size, tile_size, device=data.device),
        stride=tile_stride,
    )

    # Rotation
    pattern = rotate(pattern, rotation_degree)
    if rotation_padding != 0:  # TODO: introduce a common `crop` funciton, as this is a common bug
        pattern = pattern[
            ..., rotation_padding:-rotation_padding, rotation_padding:-rotation_padding
        ]
    return pattern


@builder.register("square_tile_pattern_aug")
@typechecked
@prob_aug
def square_tile_pattern_aug(
    data: TensorTypeVar,
    tile_size: Union[distributions.Distribution, Number],
    tile_stride: Union[distributions.Distribution, Number],
    max_brightness_change: Union[distributions.Distribution, Number],
    rotation_degree: Union[distributions.Distribution, Number] = 0.0,
    preserve_data_val: Optional[Number] = 0,
    repeats: int = 1,
    device: torch.types.Device = "cpu",
):
    assert data.shape[-1] == data.shape[-2]

    data_ = tensor_ops.to_torch(data, device=device).float()
    combined_pattern = torch.zeros_like(data_)
    max_brightness_change = (
        random.choice((-1, 1)) * distributions.to_distribution(max_brightness_change)()
    )

    for _ in range(repeats):
        # Square pattern + Rotation
        pattern = _random_square_tile_pattern(
            data_,
            tile_size=tile_size,
            tile_stride=tile_stride,
            rotation_degree=rotation_degree,
        )

        # Relative pattern brightness
        pattern *= distributions.uniform_dist(-1.0, 1.0)()

        # Translation
        w_offset = random.randint(0, pattern.shape[-1] - data_.shape[-1])
        h_offset = random.randint(0, pattern.shape[-2] - data_.shape[-2])
        combined_pattern += pattern[
            ..., h_offset : h_offset + data_.shape[-2], w_offset : w_offset + data_.shape[-1]
        ]

    # Limit accumulated brightness change
    combined_pattern = combined_pattern / combined_pattern.abs().max() * max_brightness_change

    # No tiling pattern in empty region
    if preserve_data_val is not None:
        combined_pattern[data_ == preserve_data_val] = 0.0

    data_ += combined_pattern
    result = tensor_ops.astype(data_, data)
    return result
