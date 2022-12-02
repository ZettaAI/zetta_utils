from __future__ import annotations

from typing import Literal

import attrs
import einops
import torchfields  # pylint: disable=unused-import # monkeypatch

from zetta_utils import alignment, builder, mazepa, tensor_ops
from zetta_utils.bcube import BoundingCube
from zetta_utils.layer.volumetric import (
    VolumetricIndex,
    VolumetricIndexChunker,
    VolumetricLayer,
)
from zetta_utils.typing import IntVec3D, Vec3D

from .. import build_chunked_apply_flow


@builder.register("WarpOperation")
@mazepa.taskable_operation_cls
@attrs.frozen()
class WarpOperation:
    mode: Literal["mask", "img", "field"]
    crop: IntVec3D = (0, 0, 0)
    mask_value_thr: float = 0
    # preserve_black: bool = False

    def __attrs_post_init__(self):
        if self.crop[-1] != 0:
            raise ValueError(f"Z crop must be equal to 0. Received: {self.crop}")

    def __call__(
        self,
        idx: VolumetricIndex,
        dst: VolumetricLayer,
        src: VolumetricLayer,
        field: VolumetricLayer,
    ) -> None:
        idx_padded = idx.pad(self.crop)
        field_data_raw = field[idx_padded]
        xy_translation = alignment.field_profilers.profile_field2d_percentile(field_data_raw)

        field_data_raw[0] -= xy_translation[0]
        field_data_raw[1] -= xy_translation[1]

        # TODO: big quesiton mark. In zetta_utils everythign is XYZ, so I don't understand
        # why the order is flipped here. It worked for a corgie field, so leaving it in.
        # Pls help:
        src_idx_padded = idx_padded.translate((xy_translation[1], xy_translation[0], 0))
        src_data_raw = src[src_idx_padded]

        src_data = einops.rearrange(src_data_raw, "C X Y Z -> Z C X Y")
        field_data = einops.rearrange(
            field_data_raw, "C X Y Z -> Z C X Y"
        ).field()  # type: ignore # no type for Torchfields yet

        dst_data_raw = field_data.from_pixels()(src_data.float())
        if self.mode == "mask":
            dst_data_raw = dst_data_raw > self.mask_value_thr
        dst_data_raw = dst_data_raw.to(src_data.dtype)

        # Cropping along 2 spatial dimentions, Z is batch
        dst_data_cropped = tensor_ops.crop(dst_data_raw, self.crop[:-1])
        dst_data = einops.rearrange(dst_data_cropped, "Z C X Y -> C X Y Z")
        dst[idx] = dst_data


@builder.register("build_warp_flow")
def build_warp_flow(
    bcube: BoundingCube,
    dst_resolution: Vec3D,
    chunk_size: IntVec3D,
    dst: VolumetricLayer,
    src: VolumetricLayer,
    field: VolumetricLayer,
    mode: Literal["mask", "img", "field"],
    crop: IntVec3D = (0, 0, 0),
    mask_value_thr: float = 0,
) -> mazepa.Flow:
    result = build_chunked_apply_flow(
        operation=WarpOperation(
            crop=crop, mode=mode, mask_value_thr=mask_value_thr
        ),  # type: ignore
        chunker=VolumetricIndexChunker(chunk_size=chunk_size),
        idx=VolumetricIndex(bcube=bcube, resolution=dst_resolution),
        dst=dst,  # type: ignore
        src=src,  # type: ignore
        field=field,  # type: ignore
    )
    return result
