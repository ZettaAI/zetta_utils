from __future__ import annotations

from typing import overload

import torch

from zetta_utils import alignment
from zetta_utils.geometry import Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex, VolumetricLayer


@overload
def translation_adjusted_download(
    idx: VolumetricIndex, src: VolumetricLayer, field: VolumetricLayer
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    ...


@overload
def translation_adjusted_download(
    idx: VolumetricIndex,
    src: VolumetricLayer,
    field: None,
) -> tuple[torch.Tensor, None, tuple[int, int]]:
    ...


def translation_adjusted_download(
    idx: VolumetricIndex,
    src: VolumetricLayer,
    field: VolumetricLayer | None,
) -> tuple[torch.Tensor, torch.Tensor | None, tuple[int, int]]:
    if field is not None:
        field_data = field[idx]
        xy_translation = alignment.field.profile_field2d_percentile(field_data)

        field_data[0] -= xy_translation[0]
        field_data[1] -= xy_translation[1]

        # TODO: big question mark. In zetta_utils everything is XYZ, so I don't understand
        # why the order is flipped here. It worked for a corgie field, so leaving it in.
        # Pls help:
        src_idx = idx.translated(Vec3D[int](xy_translation[1], xy_translation[0], 0))
        src_data = src[src_idx]
    else:
        field_data = None
        xy_translation = (0, 0)
        src_data = src[idx]

    return src_data, field_data, xy_translation
