from __future__ import annotations

from typing import Literal, Optional, Tuple

import einops
import torch
import torchfields
from torch.optim.lr_scheduler import ReduceLROnPlateau

from zetta_utils import builder
from zetta_utils.tensor_ops import convert
from zetta_utils.tensor_ops.generators import rand_perlin_2d_octaves
from zetta_utils.tensor_typing import TensorTypeVar


def profile_field2d_percentile(
    field: torch.Tensor,  # C, X, Y, Z
    high: float = 25,
    low: float = 75,
) -> Tuple[int, int]:

    nonzero_field_mask = (field[0] != 0) & (field[1] != 0)

    nonzero_field = field[..., nonzero_field_mask].squeeze()

    if nonzero_field.sum() == 0 or len(nonzero_field.shape) == 1:
        result = (0, 0)
    else:
        low_l = percentile(nonzero_field, low)
        high_l = percentile(nonzero_field, high)
        mid = 0.5 * (low_l + high_l)
        result = (int(mid[0]), int(mid[1]))

    return result


def percentile(field: torch.Tensor, q: float):
    """
    Return the ``q``-th percentile of the flattened input tensor's data.

    :param field: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    result = field.quantile(0.01 * float(q), dim=1, interpolation="nearest")
    return result


@builder.register("invert_field")
def invert_field(
    src: TensorTypeVar, mode: Literal["opti", "torchfields"] = "opti"
) -> TensorTypeVar:
    if not src.any():
        return src

    src_tensor = convert.to_torch(src)
    if mode == "opti":
        result = invert_field_opti(src_tensor)
    else:
        src_zcxy = (
            einops.rearrange(src_tensor, "C X Y Z -> Z C X Y").cuda().field_()  # type: ignore
        )
        with torchfields.set_identity_mapping_cache(True):
            result_zcxy = (~(src_zcxy.from_pixels())).pixels()
        result = einops.rearrange(result_zcxy, "Z C X Y -> C X Y Z")
    return convert.astype(result, src, cast=True)


def invert_field_opti(src: torch.Tensor, num_iter: int = 200, lr: float = 1e-3) -> torch.Tensor:
    if not src.any():
        return src

    src_zcxy = (
        einops.rearrange(src, "C X Y Z -> Z C X Y").cuda().field_().from_pixels()  # type: ignore
    )

    with torchfields.set_identity_mapping_cache(True):
        inverse_zcxy = (-src_zcxy).clone()
        inverse_zcxy.requires_grad = True

        optimizer = torch.optim.Adam([inverse_zcxy], lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, "min", patience=5, min_lr=1e-5)
        for _ in range(num_iter):
            loss = inverse_zcxy(src_zcxy).pixels().abs().sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

        return einops.rearrange(inverse_zcxy.detach().pixels(), "Z C X Y -> C X Y Z")


@builder.register("gen_biased_perlin_noise_field")
def gen_biased_perlin_noise_field(
    shape,
    *,
    res,
    octaves=1,
    persistence=0.5,
    field_magn_thr_px=1.0,
    max_displacement_px=None,
    device="cuda",
) -> torch.Tensor:
    """Generates a perlin noise vector field with the provided median and maximum vector length."""
    eps = 1e-7
    perlin = rand_perlin_2d_octaves(shape, res, octaves, persistence, device=device)
    warp_field = einops.rearrange(perlin, "C X Y Z -> Z C X Y").field_()  # type: ignore

    vec_length = warp_field.norm(dim=1, keepdim=True).tensor_()
    vec_length_median = torch.median(vec_length)
    vec_length_centered = vec_length - vec_length_median

    vec_length_target = torch.where(
        vec_length_centered < 0,
        vec_length_centered * field_magn_thr_px / abs(vec_length_centered.min())
        + field_magn_thr_px,
        vec_length_centered
        * (max_displacement_px - field_magn_thr_px)
        / abs(vec_length_centered.max())
        + field_magn_thr_px,
    )

    warp_field *= vec_length_target / (vec_length + eps)
    return einops.rearrange(warp_field, "Z C X Y -> C X Y Z").tensor_()


def shift_tensor(tensor: torch.Tensor, dx: int, dy: int, pad_value: float = 0) -> torch.Tensor:
    """
    Shifts the given tensor by dx and dy in the horizontal and vertical directions, respectively.

    Args:
    tensor (torch.Tensor): The input tensor.
    dx (int): The horizontal shift amount (negative for left, positive for right).
    dy (int): The vertical shift amount (negative for up, positive for down).

    Returns:
    torch.Tensor: The shifted tensor.
    """
    # Check if the tensor has at least 2 dimensions
    if tensor.dim() < 2:
        raise ValueError("Tensor must have at least 2 dimensions")

    # Roll the tensor horizontally
    shifted_tensor = torch.roll(tensor, shifts=dx, dims=-1)

    # Roll the tensor vertically
    shifted_tensor = torch.roll(shifted_tensor, shifts=dy, dims=-2)

    # Pad tensor if necessary
    if dx < 0:
        shifted_tensor[..., :, dx:] = pad_value
    elif dx > 0:
        shifted_tensor[..., :, :dx] = pad_value

    if dy < 0:
        shifted_tensor[..., dy:, :] = pad_value
    elif dy > 0:
        shifted_tensor[..., :dy, :] = pad_value

    return shifted_tensor


def get_rigidity_map_zcxy(
    field: torch.Tensor,
    power: float = 2,
    diagonal_mult: float = 1.0,
    weight_map: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Kernel on Displacement field yields change of displacement

    if field.abs().sum() == 0:
        return torch.zeros((field.shape[0], field.shape[2], field.shape[3]), device=field.device)

    batch = field.shape[0]
    diff_ker = torch.tensor(
        [
            [
                [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],  # horizontal (subtract pixel on left)
                [[0, -1, 0], [0, 1, 0], [0, 0, 0]],  # vertical (subtract pixel above)
                [[-1, 0, 0], [0, 1, 0], [0, 0, 0]],  # diagonal 1 (subtract pixel up-left)
                [[0, 0, -1], [0, 1, 0], [0, 0, 0]],  # diagonal 2 (subtract pixel up-right)
            ]
        ],
        dtype=field.dtype,
        device=field.device,
    )

    diff_ker = diff_ker.permute(1, 0, 2, 3).repeat(2, 1, 1, 1)

    # Add distance between pixel to get absolute displacement
    diff_bias = torch.tensor(
        [1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 1.0],
        dtype=field.dtype,
        device=field.device,
    )
    delta = torch.conv2d(field, diff_ker, diff_bias, groups=2, padding=[2, 2])

    # delta1 = delta.reshape(2, 4, *delta.shape[-2:]).permute(1, 2, 3, 0) # original
    delta = delta.reshape(batch, 2, 4, *delta.shape[-2:]).permute(0, 2, 3, 4, 1)

    # spring_lengths1 = torch.norm(delta1, dim=3)
    spring_lengths = torch.norm(delta, dim=-1)

    straight_len = 1
    diagonal_len = 2 ** (1 / 2)

    if weight_map is None:
        spring_defs = torch.stack(
            [
                (spring_lengths[:, 0, 1:-1, 1:-1] - straight_len),  # horizontal
                (spring_lengths[:, 0, 1:-1, 2:] - straight_len),  # horizontal
                (spring_lengths[:, 1, 1:-1, 1:-1] - straight_len),  # vertical
                (spring_lengths[:, 1, 2:, 1:-1] - straight_len),  # vertical
                (spring_lengths[:, 2, 1:-1, 1:-1] - diagonal_len)  # diagonals
                * (diagonal_mult) ** (1 / power),
                (spring_lengths[:, 2, 2:, 2:] - diagonal_len) * (diagonal_mult) ** (1 / power),
                (spring_lengths[:, 3, 1:-1, 1:-1] - diagonal_len) * (diagonal_mult) ** (1 / power),
                (spring_lengths[:, 3, 2:, 0:-2] - diagonal_len) * (diagonal_mult) ** (1 / power),
            ]
        )
    else:
        spring_defs = torch.stack(
            [
                # Center -> Left
                (spring_lengths[:, 0, 1:-1, 1:-1] - straight_len)
                * weight_map.minimum(shift_tensor(weight_map, 1, 0, pad_value=1)),
                # Right -> Center
                (spring_lengths[:, 0, 1:-1, 2:] - straight_len)
                * weight_map.minimum(shift_tensor(weight_map, -1, 0, pad_value=1)),
                # Center -> Up
                (spring_lengths[:, 1, 1:-1, 1:-1] - straight_len)
                * weight_map.minimum(shift_tensor(weight_map, 0, 1, pad_value=1)),
                # Down -> Center
                (spring_lengths[:, 1, 2:, 1:-1] - straight_len)
                * weight_map.minimum(shift_tensor(weight_map, 0, -1, pad_value=1)),
                # Center -> Up-Left
                (spring_lengths[:, 2, 1:-1, 1:-1] - diagonal_len)
                * weight_map.minimum(shift_tensor(weight_map, 1, 1, pad_value=1))
                * (diagonal_mult) ** (1 / power),
                # Down-Right -> Center
                (spring_lengths[:, 2, 2:, 2:] - diagonal_len)
                * weight_map.minimum(shift_tensor(weight_map, -1, -1, pad_value=1))
                * (diagonal_mult) ** (1 / power),
                # Center -> Up-Right
                (spring_lengths[:, 3, 1:-1, 1:-1] - diagonal_len)
                * weight_map.minimum(shift_tensor(weight_map, -1, 1, pad_value=1))
                * (diagonal_mult) ** (1 / power),
                # Down-Left -> Center
                (spring_lengths[:, 3, 2:, 0:-2] - diagonal_len)
                * weight_map.minimum(shift_tensor(weight_map, 1, -1, pad_value=1))
                * (diagonal_mult) ** (1 / power),
            ]
        )

    # Slightly faster than sum() + pow(), and obviates need for abs() if power is odd
    result = torch.norm(spring_defs, p=power, dim=0).pow(power)

    total = 4 + 4 * diagonal_mult

    result /= total

    # Remove incorrect smoothness values caused by 2px zero padding
    result[..., 0:2, :] = 0
    result[..., -2:, :] = 0
    result[..., :, 0:2] = 0
    result[..., :, -2:] = 0

    # Ensure result is a plain tensor (rather than a DisplacementField)
    if isinstance(result, torchfields.Field):
        result = result.tensor()
    return result


@builder.register("get_rigidity_map")
def get_rigidity_map(
    field: torch.Tensor, power: float = 2, diagonal_mult: float = 1.0
) -> torch.Tensor:
    field_zcxy = einops.rearrange(field, "C X Y Z -> Z C X Y")
    result_zcxy = get_rigidity_map_zcxy(
        field_zcxy, power=power, diagonal_mult=diagonal_mult
    ).unsqueeze(1)
    result = einops.rearrange(result_zcxy, "Z C X Y -> C X Y Z")
    return result
