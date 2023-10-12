from __future__ import annotations

import copy
from typing import Literal, Optional, Protocol, Sequence, Union

import attrs
import einops
import torch
import torchfields  # pylint: disable=unused-import

from zetta_utils import builder, mazepa, tensor_ops
from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric import (
    VolumetricIndex,
    VolumetricIndexTranslator,
    VolumetricLayer,
)
from zetta_utils.mazepa import semaphore
from zetta_utils.mazepa_layer_processing.alignment.common import (
    translation_adjusted_download,
)
from zetta_utils.mazepa_layer_processing.common import build_subchunkable_apply_flow


class ComputeFieldFn(Protocol):
    __name__: str

    def __call__(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_field: Optional[torch.Tensor],
    ) -> torch.Tensor:
        ...


@builder.register("ComputeFieldOperation")
@mazepa.taskable_operation_cls
@attrs.mutable
class ComputeFieldOperation:
    fn: ComputeFieldFn
    crop_pad: Sequence[int] = (0, 0, 0)
    res_change_mult: Sequence[float] = (1, 1, 1)
    output_crop_px: Sequence[int] = attrs.field(init=False)

    def get_operation_name(self) -> str:
        if hasattr(self.fn, "__name__"):
            return f"ComputeField({self.fn.__name__})"
        elif hasattr(self.fn, "name"):
            return f"ComputeField({self.fn.name})"
        else:
            return "ComputeField"

    def with_added_crop_pad(self, crop_pad: Vec3D[int]) -> ComputeFieldOperation:
        return attrs.evolve(self, crop_pad=Vec3D(*self.crop_pad) + crop_pad)

    def get_input_resolution(self, dst_resolution: Vec3D[float]) -> Vec3D[float]:
        return dst_resolution / Vec3D(*self.res_change_mult)

    def __attrs_post_init__(self):
        output_crop_px = Vec3D(*self.crop_pad) / Vec3D(*self.res_change_mult)

        for e in output_crop_px:
            if not e.is_integer():
                raise ValueError(
                    f"Destination layer crop pad of {self.crop_pad} with resolution change "
                    f"multiplier of {self.res_change_mult} results in non-integer "
                    f"output crop of {output_crop_px}."
                )
        self.output_crop_px = Vec3D[int](*(int(e) for e in output_crop_px))

    def __call__(
        self,
        idx: VolumetricIndex,
        dst: VolumetricLayer,
        src: VolumetricLayer,
        tgt: VolumetricLayer,
        src_field: Optional[VolumetricLayer],
        tgt_field: Optional[VolumetricLayer],
    ):
        with semaphore("read"):
            idx_input = copy.deepcopy(idx)
            idx_input.resolution = self.get_input_resolution(idx.resolution)
            idx_input_padded = idx_input.padded(self.crop_pad)

            src_data, src_field_data, src_translation = translation_adjusted_download(
                src=src,
                field=src_field,
                idx=idx_input_padded,
            )
        if src_data.abs().sum() > 0:
            with semaphore("read"):
                tgt_data, tgt_field_data, _ = translation_adjusted_download(
                    src=tgt, field=tgt_field, idx=idx_input_padded
                )
            with semaphore("cpu"):
                if tgt_field_data is not None:
                    tgt_field_data_zcxy = einops.rearrange(tgt_field_data, "C X Y Z -> Z C X Y")
                    tgt_data_zcxy = einops.rearrange(tgt_data, "C X Y Z -> Z C X Y")
                    tgt_nonz_zcxy = tgt_data_zcxy != 0
                    tgt_data_warped = tgt_field_data_zcxy.field().from_pixels()(  # type: ignore
                        tgt_data_zcxy.float()
                    )
                    tgt_nonz_warped = tgt_field_data_zcxy.field().from_pixels()(  # type: ignore
                        (tgt_nonz_zcxy != 0).float()
                    )
                    tgt_data_warped[tgt_nonz_warped < 0.1] = 0
                    tgt_data_final = einops.rearrange(
                        tgt_data_warped,
                        "Z C X Y -> C X Y Z",
                    )
                else:
                    tgt_data_final = tgt_data

            with semaphore("cuda"):
                result_raw = self.fn(
                    src=src_data,
                    tgt=tgt_data_final,
                    src_field=src_field_data,
                )
                result = tensor_ops.crop(result_raw, crop=self.output_crop_px)
                torch.cuda.empty_cache()

            with semaphore("write"):
                result[0] += src_translation[0]
                result[1] += src_translation[1]
                dst[idx] = result


@builder.register("ComputeFieldFlowSchema")
@mazepa.flow_schema_cls
@attrs.mutable
class ComputeFieldFlowSchema:
    operation: ComputeFieldOperation

    processing_chunk_sizes: Sequence[Sequence[int]]
    processing_crop_pads: Sequence[int] | Sequence[Sequence[int]] = (0, 0, 0)
    processing_blend_pads: Sequence[int] | Sequence[Sequence[int]] = (0, 0, 0)
    processing_blend_modes: Union[
        Literal["linear", "quadratic"], Sequence[Literal["linear", "quadratic"]]
    ] = "linear"
    level_intermediaries_dirs: Sequence[str | None] | None = None
    max_reduction_chunk_sizes: Sequence[int] | Sequence[Sequence[int]] | None = None
    expand_bbox_processing: bool = False
    shrink_processing_chunk: bool = False

    def flow(
        self,
        bbox: BBox3D,
        dst_resolution: Sequence[float],
        dst: VolumetricLayer,
        src: VolumetricLayer,
        tgt: Optional[VolumetricLayer] = None,
        src_field: Optional[VolumetricLayer] = None,
        tgt_field: Optional[VolumetricLayer] = None,
        tgt_offset: Sequence[float] = (0, 0, 0),
        src_offset: Sequence[float] = (0, 0, 0),
    ):
        if tgt is None:
            tgt = src

        tgt = copy.deepcopy(tgt)
        input_resolution = self.operation.get_input_resolution(Vec3D(*dst_resolution))
        tgt_offsetter = VolumetricIndexTranslator(offset=tgt_offset, resolution=input_resolution)
        src_offsetter = VolumetricIndexTranslator(offset=src_offset, resolution=input_resolution)
        tgt = tgt.with_procs(index_procs=(tgt_offsetter,) + tgt.index_procs)
        src = src.with_procs(index_procs=(src_offsetter,) + src.index_procs)

        if tgt_field is not None:
            tgt_field = tgt_field.with_procs(index_procs=(tgt_offsetter,) + tgt_field.index_procs)

        if src_field is not None:
            src_field = src_field.with_procs(index_procs=(src_offsetter,) + src_field.index_procs)
        cf_flow = build_subchunkable_apply_flow(
            op=self.operation,  # type: ignore
            processing_chunk_sizes=self.processing_chunk_sizes,
            processing_crop_pads=self.processing_crop_pads,
            processing_blend_pads=self.processing_blend_pads,
            processing_blend_modes=self.processing_blend_modes,
            expand_bbox_processing=self.expand_bbox_processing,
            shrink_processing_chunk=self.shrink_processing_chunk,
            max_reduction_chunk_sizes=self.max_reduction_chunk_sizes,
            level_intermediaries_dirs=self.level_intermediaries_dirs,
            skip_intermediaries=not self.level_intermediaries_dirs,
            bbox=bbox,
            dst_resolution=dst_resolution,
            dst=dst,
            op_kwargs={
                "src": src,
                "tgt": tgt,
                "src_field": src_field,
                "tgt_field": tgt_field,
            },
        )

        yield cf_flow
