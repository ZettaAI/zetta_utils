from __future__ import annotations

from typing import Literal

import attrs
import einops
import torchfields  # pylint: disable=unused-import # monkeypatch

from zetta_utils import builder, mazepa, tensor_ops
from zetta_utils.geometry import BBox3D, IntVec3D, Vec3D
from zetta_utils.layer.volumetric import (
    VolumetricIndex,
    VolumetricIndexChunker,
    VolumetricLayer,
)
from zetta_utils.mazepa_layer_processing.alignment.common import (
    translation_adjusted_download,
)

from ..common import build_chunked_apply_flow


@builder.register("WarpOperation")
@mazepa.taskable_operation_cls
@attrs.frozen()
class WarpOperation:
    mode: Literal["mask", "img", "field"]
    crop_pad: IntVec3D = IntVec3D(0, 0, 0)
    mask_value_thr: float = 0
    # preserve_black: bool = False

    def get_operation_name(self):
        return f"WarpOperation[{self.mode}]"

    def get_input_resolution(self, dst_resolution: Vec3D) -> Vec3D:  # pylint: disable=no-self-use
        return dst_resolution

    def with_added_crop_pad(self, crop_pad: IntVec3D) -> WarpOperation:
        return attrs.evolve(self, crop=self.crop_pad + crop_pad)

    def __attrs_post_init__(self):
        if self.crop_pad[-1] != 0:
            raise ValueError(f"Z crop pad must be equal to 0. Received: {self.crop_pad}")

    def __call__(  # pylint: disable=too-many-locals
        self,
        idx: VolumetricIndex,
        dst: VolumetricLayer,
        src: VolumetricLayer,
        field: VolumetricLayer,
    ) -> None:
        idx_padded = idx.padded(self.crop_pad)
        src_data_raw, field_data_raw, _ = translation_adjusted_download(
            src=src,
            field=field,
            idx=idx_padded,
        )

        src_data = einops.rearrange(src_data_raw, "C X Y Z -> Z C X Y")
        field_data = einops.rearrange(
            field_data_raw, "C X Y Z -> Z C X Y"
        ).field()  # type: ignore # no type for Torchfields yet

        dst_data_raw = field_data.from_pixels()(src_data.float())
        if self.mode == "mask":
            dst_data_raw = dst_data_raw > self.mask_value_thr
        dst_data_raw = dst_data_raw.to(src_data.dtype)

        # Cropping along 2 spatial dimentions, Z is batch
        # the typed generation is necessary because mypy cannot tell that when you slice an
        # IntVec3D, the outputs contain ints (might be due to the fact that np.int64s are not ints
        crop_2d = tuple(int(e) for e in self.crop_pad[:-1])
        dst_data_cropped = tensor_ops.crop(dst_data_raw, crop_2d)
        dst_data = einops.rearrange(dst_data_cropped, "Z C X Y -> C X Y Z")
        dst[idx] = dst_data


@builder.register("build_warp_flow")
def build_warp_flow(
    bbox: BBox3D,
    dst_resolution: Vec3D,
    chunk_size: IntVec3D,
    dst: VolumetricLayer,
    src: VolumetricLayer,
    field: VolumetricLayer,
    mode: Literal["mask", "img", "field"],
    crop_pad: IntVec3D = IntVec3D(0, 0, 0),
    mask_value_thr: float = 0,
) -> mazepa.Flow:
    result = build_chunked_apply_flow(
        operation=WarpOperation(
            crop_pad=crop_pad, mode=mode, mask_value_thr=mask_value_thr
        ),  # type: ignore
        chunker=VolumetricIndexChunker(chunk_size=chunk_size),
        idx=VolumetricIndex(bbox=bbox, resolution=dst_resolution),
        dst=dst,  # type: ignore
        src=src,  # type: ignore
        field=field,  # type: ignore
    )
    return result
