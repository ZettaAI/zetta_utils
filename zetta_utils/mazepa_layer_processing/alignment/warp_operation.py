from __future__ import annotations

from typing import Literal, Sequence

import attrs
import einops
import torchfields  # pylint: disable=unused-import # monkeypatch

from zetta_utils import builder, mazepa, tensor_ops
from zetta_utils.geometry import Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex, VolumetricLayer
from zetta_utils.mazepa_layer_processing.alignment.common import (
    translation_adjusted_download,
)


@builder.register("WarpOperation")
@mazepa.taskable_operation_cls
@attrs.frozen()
class WarpOperation:
    mode: Literal["mask", "img", "field"]
    crop_pad: Sequence[int] = (0, 0, 0)
    mask_value_thr: float = 0
    use_translation_adjustment: bool = True
    translation_granularity: int = 1

    def get_operation_name(self):
        return f"WarpOperation<{self.mode}>"

    def get_input_resolution(self, dst_resolution: Vec3D) -> Vec3D:  # pylint: disable=no-self-use
        return dst_resolution

    def with_added_crop_pad(self, crop_pad: Vec3D[int]) -> WarpOperation:
        return attrs.evolve(self, crop_pad=Vec3D(*self.crop_pad) + crop_pad)

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
        if self.use_translation_adjustment:
            src_data_raw, field_data_raw, xy_translation = translation_adjusted_download(
                src=src,
                field=field,
                idx=idx_padded,
                translation_granularity=self.translation_granularity,
            )
        else:
            src_data_raw = src[idx_padded]
            field_data_raw = field[idx_padded]
            xy_translation = (0, 0)

        src_data = einops.rearrange(src_data_raw, "C X Y Z -> Z C X Y")
        field_data = einops.rearrange(
            field_data_raw, "C X Y Z -> Z C X Y"
        ).field_()  # type: ignore # no type for Torchfields yet

        with torchfields.set_identity_mapping_cache(True):
            if self.mode == "field":
                src_data = src_data.field_().from_pixels()  # type: ignore

            dst_data_raw = field_data.from_pixels()(src_data.float())
            if self.mode == "mask":
                dst_data_raw = dst_data_raw > self.mask_value_thr
            elif self.mode == "field":
                dst_data_raw = dst_data_raw.pixels()
                dst_data_raw.x += xy_translation[0]
                dst_data_raw.y += xy_translation[1]

        dst_data_raw = dst_data_raw.to(src_data.dtype)

        # Cropping along 2 spatial dimentions, Z is batch
        # the typed generation is necessary because mypy cannot tell that when you slice an
        # Vec3D[int], the outputs contain ints (might be due to the fact that np.int64s
        # are not ints
        crop_2d = tuple(int(e) for e in self.crop_pad[:-1])
        dst_data_cropped = tensor_ops.crop(dst_data_raw, crop_2d)
        dst_data = einops.rearrange(dst_data_cropped, "Z C X Y -> C X Y Z")
        dst[idx] = dst_data
