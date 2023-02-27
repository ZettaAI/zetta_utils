from __future__ import annotations

import math
import random
from typing import Callable, Optional, Sequence, Union

import torch
from torchvision.transforms.functional import rotate
from typeguard import typechecked

from zetta_utils import builder, distributions, tensor_ops
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
        combined_pattern = combined_pattern / combined_pattern.abs().max() * max_brightness_change

        # No tiling pattern in empty region
        if preserve_data_val is not None:
            combined_pattern[section_torch == preserve_data_val] = 0.0

        section_torch += combined_pattern

    result = tensor_ops.astype(data_torch_3d, data)
    if len(data.shape) == 3:
        result = result.squeeze(-1)
    return result


# https://gist.github.com/vadimkantorov/ac1b097753f217c5c11bc2ff396e0a57
@builder.register("rand_perlin_2d")
@typechecked
def rand_perlin_2d(
    shape: Sequence[int],  # CXYZ
    res: Sequence[int],
    fade: Callable[[torch.Tensor], torch.Tensor] = lambda t: 6 * t ** 5
    - 15 * t ** 4
    + 10 * t ** 3,
    device: torch.types.Device = None,
) -> torch.Tensor:
    if len(shape) != 4:
        raise ValueError(f"'shape' expected length 4 (CXYZ), got {len(shape)}")
    if len(res) != 2:
        raise ValueError(f"'res' expected length 2, got {len(res)}")

    delta = (res[0] / shape[1], res[1] / shape[2])
    tiles = (shape[1] // res[0], shape[2] // res[1])

    grid = (
        torch.stack(
            torch.meshgrid(
                torch.arange(0, res[0], delta[0], device=device),
                torch.arange(0, res[1], delta[1], device=device),
                indexing="ij",
            ),
            dim=-1,
        )
        % 1
    )

    angles = (
        2 * math.pi * torch.rand(shape[-1], shape[0], res[0] + 1, res[1] + 1, device=device)
    )  # ZCXY
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    tile_grads: Callable[[slice, slice], torch.Tensor] = (
        lambda slice1, slice2: gradients[..., slice1, slice2, :]
        .repeat_interleave(tiles[0], -3)
        .repeat_interleave(tiles[1], -2)
    )
    dot = lambda grad, shift: (
        torch.stack(
            (
                grid[: shape[1], : shape[2], 0] + shift[0],
                grid[: shape[1], : shape[2], 1] + shift[1],
            ),
            dim=-1,
        )
        * grad[..., : shape[1], : shape[2], :]
    ).sum(dim=-1)

    n00 = dot(tile_grads(slice(0, -1), slice(0, -1)), [0, 0])
    n10 = dot(tile_grads(slice(1, None), slice(0, -1)), [-1, 0])
    n01 = dot(tile_grads(slice(0, -1), slice(1, None)), [0, -1])
    n11 = dot(tile_grads(slice(1, None), slice(1, None)), [-1, -1])
    weights = fade(grid[: shape[1], : shape[2]])
    return math.sqrt(2) * torch.lerp(
        torch.lerp(n00, n10, weights[..., 0]),
        torch.lerp(n01, n11, weights[..., 0]),
        weights[..., 1],
    ).permute(
        (1, 2, 3, 0)
    )  # CXYZ


@builder.register("rand_perlin_2d_octaves")
@typechecked
def rand_perlin_2d_octaves(
    shape: Sequence[int],
    res: Sequence[int],
    octaves: int = 1,
    persistence: float = 0.5,
    device: torch.types.Device = None,
) -> torch.Tensor:
    if len(shape) != 4:
        raise ValueError(f"'shape' expected length 4 (CXYZ), got {len(shape)}")
    if len(res) != 2:
        raise ValueError(f"'res' expected length 2, got {len(res)}")

    noise = torch.zeros(*shape, device=device)
    frequency = 1
    amplitude = 1.0
    for _ in range(octaves):
        noise += amplitude * rand_perlin_2d(
            shape,
            (frequency * res[0], frequency * res[1]),
            device=device,
        )
        frequency *= 2
        amplitude *= persistence
    return noise
