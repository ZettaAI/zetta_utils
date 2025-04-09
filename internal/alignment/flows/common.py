from __future__ import annotations

from typing import cast, overload

import einops
import torch

from zetta_utils.geometry import Vec3D
from zetta_utils.internal import alignment
from zetta_utils.layer.volumetric import VolumetricIndex, VolumetricLayer
from zetta_utils.tensor_ops import convert


@overload
def translation_adjusted_download(
    idx: VolumetricIndex,
    src: VolumetricLayer,
    field: VolumetricLayer,
    mask: VolumetricLayer,
    translation_granularity: int = ...,
    device: str | torch.device | None = ...,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
    ...


@overload
def translation_adjusted_download(
    idx: VolumetricIndex,
    src: VolumetricLayer,
    field: VolumetricLayer,
    mask: None = ...,
    translation_granularity: int = ...,
    device: str | torch.device | None = ...,
) -> tuple[torch.Tensor, torch.Tensor, None, tuple[int, int]]:
    ...


@overload
def translation_adjusted_download(
    idx: VolumetricIndex,
    src: VolumetricLayer,
    field: None,
    mask: VolumetricLayer,
    translation_granularity: int = ...,
    device: str | torch.device | None = ...,
) -> tuple[torch.Tensor, None, torch.Tensor, tuple[int, int]]:
    ...


@overload
def translation_adjusted_download(
    idx: VolumetricIndex,
    src: VolumetricLayer,
    field: None,
    mask: None = ...,
    translation_granularity: int = ...,
    device: str | torch.device | None = ...,
) -> tuple[torch.Tensor, None, None, tuple[int, int]]:
    ...


def translation_adjusted_download(
    idx: VolumetricIndex,
    src: VolumetricLayer,
    field: VolumetricLayer | None,
    mask: VolumetricLayer | None = None,
    translation_granularity: int = 1,
    device: str | torch.device | None = None,
):
    field_data = None
    mask_data = None
    if field is not None:
        field_data = convert.to_torch(field[idx], device=device)
        xy_translation_raw = alignment.field.profile_field2d_percentile(field_data)
        xy_translation = cast(
            tuple[int, int],
            tuple(
                translation_granularity * round(e / translation_granularity)
                for e in xy_translation_raw
            ),
        )

        field_data[0] -= xy_translation[0]
        field_data[1] -= xy_translation[1]

        # TODO: big question mark. In zetta_utils everything is XYZ, so I don't understand
        # why the order is flipped here. It worked for a corgie field, so leaving it in.
        # Pls help:
        src_idx = idx.translated(Vec3D[int](xy_translation[1], xy_translation[0], 0))
        src_data = convert.to_torch(src[src_idx], device=device)
        if mask is not None:
            mask_data = convert.to_torch(mask[src_idx], device=device)
    else:
        xy_translation = (0, 0)
        src_data = convert.to_torch(src[idx], device=device)
        if mask is not None:
            mask_data = convert.to_torch(mask[idx], device=device)

    return src_data, field_data, mask_data, xy_translation


def warp_preserve_zero(
    data_cxyz: torch.Tensor, field_cxyz: torch.Tensor | None, preserve_zero: bool = True
) -> torch.Tensor:
    if field_cxyz is None:
        return data_cxyz

    field_zcxy = (
        einops.rearrange(field_cxyz, "C X Y Z -> Z C X Y").field_().from_pixels()  # type: ignore
    )
    data_zcxy = einops.rearrange(data_cxyz, "C X Y Z -> Z C X Y")
    data_warped_zcxy = field_zcxy(data_zcxy.float())

    if preserve_zero:
        zeros_warped = field_zcxy((data_zcxy == 0).float()) > 0.1
        data_warped_zcxy[zeros_warped] = 0

    data_warped_cxyz = einops.rearrange(data_warped_zcxy, "Z C X Y -> C X Y Z")
    return data_warped_cxyz
