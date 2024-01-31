import math
from typing import Callable, Sequence

import affine
import einops
import torch
import torchfields  # pylint: disable=unused-import
from typeguard import typechecked

from zetta_utils import builder


@builder.register("get_affine_field")
def get_affine_field(
    size,
    trans_x_px=0,
    trans_y_px=0,
    rot_deg=0,
    shear_x_deg=0,
    shear_y_deg=0,
    scale=1,
) -> torch.Tensor:
    """
    Return 2D displacement field that represents the given affine transformation.
    Transformations are applied in the following order -- translation->rotation->shear->scale.
    Note that the resulting field is represented in pixel magnitudes.

    :param size: Shape along the X and Y dimension of the resulting field.
    :param trans_x_px: X translation in pixels, from left to right.
    :param trans_y_px: Y translation in pixels, from top to bottom.
    :param rot_deg: Rotation degrees, clockwise
    :param shear_x_deg: X shear degrees.
    :param shear_y_deg: Y shear degrees.
    :return: The torch tensor in CXYZ.
    """
    aff = (
        affine.Affine.translation(-trans_x_px * 2 / size, -trans_y_px * 2 / size)
        * affine.Affine.rotation(-rot_deg)
        * affine.Affine.shear(-shear_x_deg, -shear_y_deg)
        * affine.Affine.scale(1 / scale)
    )
    mat = torch.tensor([[aff.a, aff.b, aff.c], [aff.d, aff.e, aff.f]]).unsqueeze(0)
    field = torch.Field.affine_field(mat, size=(1, 2, size, size))  # type: ignore
    return einops.rearrange(field, "Z C X Y -> C X Y Z")


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
        2
        * math.pi
        * torch.rand(
            shape[-1],
            shape[0],
            res[0] + 1,
            res[1] + 1,
            device=device,
        )
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
