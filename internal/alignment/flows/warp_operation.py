from __future__ import annotations

from typing import Literal, Sequence

import attrs
import einops
import torch
import torchfields  # pylint: disable=unused-import # monkeypatch

from zetta_utils import builder, mazepa, tensor_ops
from zetta_utils.geometry import Vec3D
from zetta_utils.internal.alignment.flows.common import translation_adjusted_download
from zetta_utils.layer.volumetric import VolumetricIndex, VolumetricLayer
from zetta_utils.mazepa import semaphore
from zetta_utils.mazepa.semaphores import SemaphoreType
from zetta_utils.tensor_ops import convert


@builder.register("WarpOperation")
@mazepa.taskable_operation_cls
@attrs.frozen()
class WarpOperation:
    mode: Literal["mask", "img", "field"]
    crop_pad: Sequence[int] = (0, 0, 0)
    mask_value_thr: float = 0
    use_translation_adjustment: bool = True
    translation_granularity: int = 1
    downsampling_factor: Sequence[int] = (1, 1, 1)

    """
    Warp operation.

    :param downsampling_factor: apply downsampling in xyz after warping. Values
        greater than 1 will make the operator read higher resolution data, warp
        it, then downsample to the requested resolution.
    """

    def get_operation_name(self):
        return f"WarpOperation<{self.mode}>"

    def get_input_resolution(self, dst_resolution: Vec3D[float]) -> Vec3D[float]:
        return dst_resolution / Vec3D(*self.downsampling_factor)

    def with_added_crop_pad(self, crop_pad: Vec3D[int]) -> WarpOperation:
        return attrs.evolve(self, crop_pad=Vec3D(*self.crop_pad) + crop_pad)

    def __attrs_post_init__(self):
        if self.crop_pad[-1] != 0:
            raise ValueError(f"Z crop pad must be equal to 0. Received: {self.crop_pad}")

    def __call__(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        self,
        idx: VolumetricIndex,
        dst: VolumetricLayer,
        src: VolumetricLayer,
        field: VolumetricLayer,
        mask: VolumetricLayer | None = None,
    ) -> None:
        semaphore_device: SemaphoreType = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(semaphore_device)

        idx_padded = idx.padded(self.crop_pad)
        idx_padded.resolution = self.get_input_resolution(idx_padded.resolution)

        with semaphore("read"):
            dst_original_data = None
            if mask is not None:
                dst_original_data = convert.to_torch(dst[idx], device=device)

            # "read" even if device is CUDA - download is still I/O bound
            if self.use_translation_adjustment:
                (
                    src_data_raw,
                    field_data_raw,
                    mask_data_raw,
                    xy_translation,
                ) = translation_adjusted_download(
                    src=src,
                    field=field,
                    mask=mask,
                    idx=idx_padded,
                    translation_granularity=self.translation_granularity,
                    device=device,
                )
            else:
                src_data_raw = convert.to_torch(src[idx_padded], device=device)
                field_data_raw = convert.to_torch(field[idx_padded], device=device)
                mask_data_raw = None
                if mask is not None:
                    mask_data_raw = convert.to_torch(mask[idx_padded], device=device)
                xy_translation = (0, 0)

        with semaphore(semaphore_device):
            mask_data = None
            if mask is not None:
                assert mask_data_raw is not None
                mask_data = einops.rearrange(mask_data_raw, "C X Y Z -> Z C X Y")
            src_data = einops.rearrange(src_data_raw, "C X Y Z -> Z C X Y")
            field_data = einops.rearrange(
                field_data_raw, "C X Y Z -> Z C X Y"
            ).field_()  # type: ignore # no type for Torchfields yet

            with torchfields.set_identity_mapping_cache(True):
                if self.mode == "field":
                    if self.downsampling_factor != (1, 1, 1):
                        raise NotImplementedError(
                            "Downsampling should work but not tested with field warping"
                        )
                    src_data = src_data.field_().from_pixels()  # type: ignore

                dst_data_raw = field_data.from_pixels()(src_data.float())

                mask_data_warped = None
                if mask is not None:
                    assert mask_data is not None
                    mask_data_warped = (
                        field_data.from_pixels()(mask_data.float()) > self.mask_value_thr
                    )

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
            crop_2d = tuple(
                int(e * mult) for e, mult in zip(self.crop_pad[:-1], self.downsampling_factor[:-1])
            )
            dst_data_cropped = tensor_ops.crop(dst_data_raw, crop_2d)
            dst_data = einops.rearrange(dst_data_cropped, "Z C X Y -> C X Y Z")

            dst_mask = None
            if mask is not None:
                assert mask_data_warped is not None
                mask_data_cropped = tensor_ops.crop(mask_data_warped, crop_2d)
                dst_mask = einops.rearrange(mask_data_cropped, "Z C X Y -> C X Y Z")

            if self.downsampling_factor != (1, 1, 1):
                dst_data = tensor_ops.interpolate(
                    data=dst_data,
                    scale_factor=[1.0 / k for k in self.downsampling_factor],
                    mode=self.mode,
                )

            if dst_mask is not None:
                assert dst_mask is not None
                assert dst_original_data is not None
                dst_data[dst_mask == 0] = dst_original_data[dst_mask == 0]

        with semaphore("write"):
            dst[idx] = dst_data
