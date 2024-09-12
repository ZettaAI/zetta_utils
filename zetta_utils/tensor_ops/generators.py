import math
from typing import Callable, Iterable, Sequence

import affine
import einops
import torch
import torchfields  # pylint: disable=unused-import
from neuroglancer.viewer_state import LineAnnotation
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.db_annotations.annotation import AnnotationDBEntry
from zetta_utils.geometry.vec import VEC3D_PRECISION, Vec3D
from zetta_utils.layer.volumetric.index import VolumetricIndex
from zetta_utils.tensor_ops import convert
from zetta_utils.tensor_typing import Tensor


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


def get_field_from_matrix(
    mat: Tensor, size: int | Sequence[int], device=None
) -> torchfields.Field:
    """
    Returns a 2D displacement field for each affine or perspective transformation matrix.

    :param mat:  [Nx]2x3 (affine) or [Nx]3x3 (perspective) ndarray or torch.Tensor.
                 The matrices defining the transformations.
    :param size: an `int`, a `tuple` or a `torch.Size` of the form `(H, W)`.
    :return:     DisplacementField for the given transformation of size `(N, 2, H, W)`

    Note:
    The matrix defines the transformation that warps the destination to the source,
    such that, \vec{x_s} = P \vec{x_d} where x_s is a point in the source image, x_d a point in
    the destination image, and P is the perspective matrix.
    The field returned will be defined over the destination image. So the matrix P should define
    the location in the source image that contribute to a pixel in the destination image.
    """

    mat = convert.to_torch(mat, device=device)
    if mat.ndimension() == 2:
        mat = mat.unsqueeze(0)
        N = 1
    elif mat.ndimension() == 3:
        N = mat.shape[0]
    else:
        raise ValueError(f"Expected 2 or 3-dimensional matrix. Received shape {mat.shape}.")

    if isinstance(size, int):
        size = [N, 2, size, size]
    elif len(size) == 2:
        size = [N, 2, size[0], size[1]]
    else:
        raise ValueError(f"Expected size to be int or a Sequence of length 2. Received {size}.")

    # Generate a grid of coordinates to apply the perspective transform
    grid_identity = torch.nn.functional.affine_grid(
        torch.eye(2, 3, device=device, dtype=mat.dtype).unsqueeze(0).expand(N, 2, 3),
        size,
        align_corners=False,
    )
    grid_identity = grid_identity.view(N, size[2], size[3], 2)

    if mat.shape[1] == 2:
        # Affine - can just use torch's affine_grid
        warped_grid = torch.nn.functional.affine_grid(mat, size, align_corners=False)
    else:
        # Perspective - need to apply the perspective transformation manually
        grid_homogeneous = torch.cat(
            [grid_identity, torch.ones(N, size[2], size[3], 1, device=device, dtype=mat.dtype)],
            dim=-1,
        )
        grid_homogeneous = grid_homogeneous.view(N, -1, 3).transpose(1, 2)

        # Apply the perspective transformation
        warped_grid_homogeneous = (
            torch.bmm(mat, grid_homogeneous).transpose(1, 2).view(N, size[2], size[3], 3)
        )
        warped_grid = warped_grid_homogeneous[..., :2] / warped_grid_homogeneous[..., 2:]

    displacement_field = warped_grid - grid_identity
    displacement_field = displacement_field.permute(0, 3, 1, 2).field_()  # type: ignore

    return displacement_field


def get_field_from_annotations(
    line_annotations: Iterable[AnnotationDBEntry],
    index: VolumetricIndex,
    device: torch.types.Device | None = None,
) -> torchfields.Field:
    """
    Returns a sparse 2D displacement field based on the provided line annotations.

    :param line_annotations: Iterable line annotations.
    :param index: VolumetricIndex specifying bounds and resolution of returned field.
    :param device: Device to use for returned field.
    :return: DisplacementField for the given line annotations. Unspecified values are NaN.
    """
    sparse_field = torch.full((1, 2, index.shape[0], index.shape[1]), torch.nan, device=device)
    contrib_sum = torch.zeros(sparse_field.shape[-2:], device=device)
    for line in line_annotations:
        annotation = line.ng_annotation
        if not isinstance(annotation, LineAnnotation):
            raise ValueError(f"Expected LineAnnotation, got {type(annotation)}")

        pointA: Vec3D = Vec3D(*annotation.pointA) / index.resolution
        pointB: Vec3D = Vec3D(*annotation.pointB) / index.resolution
        if index.contains(round(pointA, VEC3D_PRECISION)):
            x = math.floor(pointA.x) - index.start[0]
            y = math.floor(pointA.y) - index.start[1]
            # Contribution is 1 - sqrt(2) at corner of pixel, 1 at center
            # Maybe should consider adjacent pixel, but hopefully good enough for now
            contrib = 1.0 - math.sqrt((pointA.x % 1 - 0.5) ** 2 + (pointA.y % 1 - 0.5) ** 2)
            contrib_sum[x, y] += contrib

            if sparse_field[0, :, x, y].isnan().all():
                sparse_field[0, :, x, y] = 0.0

            # Field xy is flipped
            sparse_field[0, :, x, y] += contrib * torch.tensor(pointB - pointA)[:2].flipud()

    return sparse_field / contrib_sum


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

    def _dot(grad, shift):
        return (
            torch.stack(
                (
                    grid[: shape[1], : shape[2], 0] + shift[0],
                    grid[: shape[1], : shape[2], 1] + shift[1],
                ),
                dim=-1,
            )
            * grad[..., : shape[1], : shape[2], :]
        ).sum(dim=-1)

    n00 = _dot(tile_grads(slice(0, -1), slice(0, -1)), [0, 0])
    n10 = _dot(tile_grads(slice(1, None), slice(0, -1)), [-1, 0])
    n01 = _dot(tile_grads(slice(0, -1), slice(1, None)), [0, -1])
    n11 = _dot(tile_grads(slice(1, None), slice(1, None)), [-1, -1])
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
