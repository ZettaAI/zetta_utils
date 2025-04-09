from __future__ import annotations

import copy
from typing import Literal, Optional, Protocol, Sequence, Union

import attrs
import torch
import torchfields  # pylint: disable=unused-import

from zetta_utils import builder, mazepa, tensor_ops
from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.internal.alignment.flows.common import (
    translation_adjusted_download,
    warp_preserve_zero,
)
from zetta_utils.layer.volumetric import (
    VolumetricIndex,
    VolumetricIndexTranslator,
    VolumetricLayer,
)
from zetta_utils.mazepa import semaphore
from zetta_utils.mazepa.semaphores import SemaphoreType
from zetta_utils.mazepa_layer_processing.common import build_subchunkable_apply_flow


class ComputeFieldFn(Protocol):
    __name__: str

    def __call__(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_field: Optional[torch.Tensor],
        src_rig_weight: Optional[torch.Tensor],
        tgt_rig_weight: Optional[torch.Tensor],
        src_mse_weight: Optional[torch.Tensor],
        tgt_mse_weight: Optional[torch.Tensor],
        src_pinned_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        ...


@builder.register("ComputeFieldOperation")
@mazepa.taskable_operation_cls
@attrs.mutable
class ComputeFieldOperation:
    fn: ComputeFieldFn
    crop_pad: Sequence[int] = (0, 0, 0)
    res_change_mult: Sequence[float] = (1, 1, 1)
    translation_granularity: int = 1
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

    def __call__(  # pylint: disable=too-many-locals
        self,
        idx: VolumetricIndex,
        dst: VolumetricLayer,
        src: VolumetricLayer,
        tgt: VolumetricLayer,
        src_field: Optional[VolumetricLayer],
        tgt_field: Optional[VolumetricLayer],
        src_rig_weight: Optional[VolumetricLayer],
        tgt_rig_weight: Optional[VolumetricLayer],
        src_mse_weight: Optional[VolumetricLayer],
        tgt_mse_weight: Optional[VolumetricLayer],
        src_pinned_field: Optional[VolumetricLayer],
    ):
        semaphore_device: SemaphoreType = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(semaphore_device)

        with semaphore("read"):
            # "read" even if device is CUDA - download is still I/O bound
            idx_input = copy.deepcopy(idx)
            idx_input.resolution = self.get_input_resolution(idx.resolution)
            idx_input_padded = idx_input.padded(self.crop_pad)

            src_data, src_field_data, _, src_translation = translation_adjusted_download(
                src=src,
                field=src_field,
                idx=idx_input_padded,
                device=device,
                translation_granularity=self.translation_granularity,
            )
            if not src_data.any():
                return

            tgt_data, tgt_field_data, _, _ = translation_adjusted_download(
                src=tgt,
                field=tgt_field,
                idx=idx_input_padded,
                device=device,
                translation_granularity=self.translation_granularity,
            )

            def optional_translation_adjusted_download(
                weight_layer: Optional[VolumetricLayer], field_layer: Optional[VolumetricLayer]
            ) -> Optional[torch.Tensor]:
                if weight_layer is not None:
                    weight_data, _, _, _ = translation_adjusted_download(
                        src=weight_layer,
                        field=field_layer,
                        idx=idx_input_padded,
                        device=device,
                        translation_granularity=self.translation_granularity,
                    )
                else:
                    weight_data = None
                return weight_data

            src_rig_weight_data = optional_translation_adjusted_download(src_rig_weight, src_field)
            tgt_rig_weight_data = optional_translation_adjusted_download(tgt_rig_weight, tgt_field)
            src_mse_weight_data = optional_translation_adjusted_download(src_mse_weight, src_field)
            tgt_mse_weight_data = optional_translation_adjusted_download(tgt_mse_weight, tgt_field)

            src_pinned_field_data = optional_translation_adjusted_download(
                src_pinned_field, src_field
            )
            if src_pinned_field_data is not None:
                # If src_field was adjusted, also need to adjust the pinned field
                src_pinned_field_data[0] -= src_translation[0]
                src_pinned_field_data[1] -= src_translation[1]

                # Create mask to keep track of pinned vectors
                src_pinned_mask = ~(src_pinned_field_data[:1].isnan())

                # Create/update initial vector field
                if src_field_data is None:
                    src_field_data = torch.nan_to_num(src_pinned_field_data, 0.0)
                else:
                    src_field_data = torch.where(
                        src_pinned_mask, src_pinned_field_data, src_field_data
                    )
            else:
                src_pinned_mask = None

        with semaphore(semaphore_device):
            tgt_data = warp_preserve_zero(tgt_data, tgt_field_data)

            if tgt_rig_weight_data is not None:
                tgt_rig_weight_data = warp_preserve_zero(
                    tgt_rig_weight_data, tgt_field_data, preserve_zero=False
                )

            if tgt_mse_weight_data is not None:
                tgt_mse_weight_data = warp_preserve_zero(
                    tgt_mse_weight_data, tgt_field_data, preserve_zero=False
                )

            result_raw = self.fn(
                src=src_data,
                tgt=tgt_data,
                src_field=src_field_data,
                src_rig_weight=src_rig_weight_data,
                tgt_rig_weight=tgt_rig_weight_data,
                src_mse_weight=src_mse_weight_data,
                tgt_mse_weight=tgt_mse_weight_data,
                src_pinned_mask=src_pinned_mask,
            )
            result = tensor_ops.crop(result_raw, crop=self.output_crop_px)
            if semaphore_device == "cuda":
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
    max_reduction_chunk_size: Sequence[int] | None = None
    expand_bbox_resolution: bool = False
    expand_bbox_backend: bool = False
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
        src_rig_weight: VolumetricLayer | None = None,
        tgt_rig_weight: VolumetricLayer | None = None,
        src_mse_weight: VolumetricLayer | None = None,
        tgt_mse_weight: VolumetricLayer | None = None,
        src_pinned_field: VolumetricLayer | None = None,
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

        if src_rig_weight is not None:
            src_rig_weight = src_rig_weight.with_procs(
                index_procs=(src_offsetter,) + src_rig_weight.index_procs
            )

        if tgt_rig_weight is not None:
            tgt_rig_weight = tgt_rig_weight.with_procs(
                index_procs=(tgt_offsetter,) + tgt_rig_weight.index_procs
            )

        if src_mse_weight is not None:
            src_mse_weight = src_mse_weight.with_procs(
                index_procs=(src_offsetter,) + src_mse_weight.index_procs
            )

        if tgt_mse_weight is not None:
            tgt_mse_weight = tgt_mse_weight.with_procs(
                index_procs=(tgt_offsetter,) + tgt_mse_weight.index_procs
            )

        if src_pinned_field is not None:
            src_pinned_field = src_pinned_field.with_procs(
                index_procs=(src_offsetter,) + src_pinned_field.index_procs
            )

        cf_flow = build_subchunkable_apply_flow(
            op=self.operation,
            processing_chunk_sizes=self.processing_chunk_sizes,
            processing_crop_pads=self.processing_crop_pads,
            processing_blend_pads=self.processing_blend_pads,
            processing_blend_modes=self.processing_blend_modes,
            expand_bbox_resolution=self.expand_bbox_resolution,
            expand_bbox_backend=self.expand_bbox_backend,
            expand_bbox_processing=self.expand_bbox_processing,
            shrink_processing_chunk=self.shrink_processing_chunk,
            max_reduction_chunk_size=self.max_reduction_chunk_size,
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
                "src_rig_weight": src_rig_weight,
                "tgt_rig_weight": tgt_rig_weight,
                "src_mse_weight": src_mse_weight,
                "tgt_mse_weight": tgt_mse_weight,
                "src_pinned_field": src_pinned_field,
            },
        )

        yield cf_flow
