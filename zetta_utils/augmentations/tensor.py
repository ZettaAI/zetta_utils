from __future__ import annotations

import math
import random
from typing import Callable, Optional, Union, overload

import torch
from torchvision.transforms.functional import rotate
from typeguard import typechecked

from zetta_utils import builder, distributions, tensor_ops
from zetta_utils.tensor_ops.common import clone
from zetta_utils.tensor_ops.convert import astype
from zetta_utils.tensor_typing import Tensor, TensorTypeVar

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
@prob_aug
@typechecked
def add_scalar_aug(
    data: TensorTypeVar,
    value_distr: Union[distributions.Distribution, float],
    mask_fn: Optional[Callable[..., Tensor]] = None,
) -> TensorTypeVar:
    data_torch = tensor_ops.to_torch(data).float()
    weights_mask = _get_weights_mask(data_torch, mask_fn).float()
    value = distributions.to_distribution(value_distr)() * weights_mask
    data_torch += value
    result = tensor_ops.astype(data_torch, data)
    return result


@builder.register("clamp_values_aug")
@prob_aug
@typechecked
def clamp_values_aug(
    data: TensorTypeVar,
    low_distr: Optional[Union[distributions.Distribution, float]] = None,
    high_distr: Optional[Union[distributions.Distribution, float]] = None,
    mask_fn: Optional[Callable[..., Tensor]] = None,
) -> TensorTypeVar:
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
    tile_stride: Union[distributions.Distribution, float],
    tile_size: Union[distributions.Distribution, float],
    rotation_degree: Union[distributions.Distribution, float],
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
    return pattern.squeeze(0)


@builder.register("square_tile_pattern_aug")
@prob_aug
@typechecked
def square_tile_pattern_aug(  # pylint: disable=too-many-locals
    data: TensorTypeVar,
    tile_size: Union[distributions.Distribution, float],
    tile_stride: Union[distributions.Distribution, float],
    max_brightness_change: Union[distributions.Distribution, float],
    rotation_degree: Union[distributions.Distribution, float] = 0.0,
    preserve_data_val: Optional[float] = 0,
    repeats: int = 1,
    device: torch.types.Device = "cpu",
) -> TensorTypeVar:
    # C X Y (Z)
    assert data.shape[1] == data.shape[2]
    data_torch = tensor_ops.to_torch(data, device=device).float()
    if len(data.shape) == 3:
        data_torch_3d = data_torch.unsqueeze(-1)
    else:
        data_torch_3d = data_torch

    for z in range(data_torch_3d.shape[-1]):
        section_torch = data_torch_3d[..., z]
        combined_pattern = torch.zeros_like(section_torch)
        max_brightness_change = (
            random.choice((-1, 1)) * distributions.to_distribution(max_brightness_change)()
        )

        for _ in range(repeats):
            # Square pattern + Rotation
            pattern = _random_square_tile_pattern(
                section_torch,
                tile_size=tile_size,
                tile_stride=tile_stride,
                rotation_degree=rotation_degree,
            )

            # Relative pattern brightness
            pattern *= distributions.uniform_distr(-1.0, 1.0)()

            # Translation
            w_offset = random.randint(0, pattern.shape[-1] - section_torch.shape[-1])
            h_offset = random.randint(0, pattern.shape[-2] - section_torch.shape[-2])
            combined_pattern += pattern[
                ...,
                h_offset : h_offset + section_torch.shape[-2],
                w_offset : w_offset + section_torch.shape[-1],
            ]

        # Limit accumulated brightness change
        if combined_pattern.abs().max() > 1e-6:
            combined_pattern = (
                combined_pattern / combined_pattern.abs().max() * max_brightness_change
            )

        # No tiling pattern in empty region
        if preserve_data_val is not None:
            combined_pattern[section_torch == preserve_data_val] = 0.0

        section_torch += combined_pattern

    result = tensor_ops.astype(data_torch_3d, data)
    if len(data.shape) == 3:
        result = result.squeeze(-1)
    return result


@builder.register("apply_to_random_sections")
@prob_aug
@typechecked
def apply_to_random_sections(
    data: TensorTypeVar,
    fn: Callable[[TensorTypeVar], TensorTypeVar],
    num_sections: Union[distributions.Distribution, int] = 1,
) -> TensorTypeVar:
    assert data.ndim == 4, "Input must be CXYZ tensor"
    num_sections_chosen = int(distributions.to_distribution(num_sections)())
    chosen_sections = random.sample(range(0, data.shape[-1]), num_sections_chosen)
    result = clone(data)
    for i in chosen_sections:
        processed = astype(fn(data[..., i]), result)
        assert isinstance(processed, type(result))
        result[..., i] = processed  # type: ignore # mypy bug
    return result


@overload
def apply_to_random_boxes(
    data: TensorTypeVar,
    fn: Callable[[TensorTypeVar], TensorTypeVar],
    box_size: Union[distributions.Distribution, float],
    num_boxes: Union[distributions.Distribution, int] = ...,
    density: None = ...,
    vary_box_sizes: bool = ...,
    allow_partial_boxes: bool = ...,
) -> TensorTypeVar:
    ...


@overload
def apply_to_random_boxes(
    data: TensorTypeVar,
    fn: Callable[[TensorTypeVar], TensorTypeVar],
    box_size: Union[distributions.Distribution, float],
    num_boxes: None = ...,
    density: Union[distributions.Distribution, float] = ...,
    vary_box_sizes: bool = ...,
    allow_partial_boxes: bool = ...,
) -> TensorTypeVar:
    ...


@builder.register("apply_to_random_boxes")
@prob_aug
@typechecked
def apply_to_random_boxes(
    data,
    fn,
    box_size,
    num_boxes=1,
    density=None,
    vary_box_sizes=True,
    allow_partial_boxes=True,
):
    assert data.ndim == 4, "Input must be CXYZ tensor"
    result = data.clone()

    chosen_regions: list[tuple[slice, ...]] = []
    box_size_distr = distributions.to_distribution(box_size)

    def _get_box_size() -> tuple[int, ...]:
        result = tuple(int(box_size_distr() * data.shape[i + 1]) for i in range(3))
        assert min(*result) > 0
        return result

    box_size = _get_box_size()

    def _choose_box() -> tuple[slice, ...]:
        if vary_box_sizes:
            box_size = _get_box_size()
        if allow_partial_boxes:
            box_start = [random.choice(range(0, data.shape[i + 1])) for i in range(3)]
        else:
            box_start = [
                random.choice(range(0, data.shape[i + 1] - box_size[i])) for i in range(3)
            ]
        result = tuple(
            [slice(None, None)]
            + [slice(box_start[i], box_start[i] + box_size[i]) for i in range(3)]
        )
        return result

    if num_boxes is not None:
        assert density is None
        num_boxes_chosen = int(distributions.to_distribution(num_boxes)())
        for i in range(num_boxes_chosen):
            this_region = _choose_box()
            chosen_regions.append(this_region)
    else:
        assert density is not None
        chosen_density = distributions.to_distribution(density)()
        current_density = 0
        total_px = math.prod(data.shape)

        while current_density < chosen_density:
            this_region = _choose_box()
            chosen_regions.append(this_region)
            this_region_px = math.prod(
                this_region[i + 1].stop - this_region[i + 1].start for i in range(3)
            )
            current_density += this_region_px / total_px

    for region in chosen_regions:
        result[region] = fn(data[region])

    return result
