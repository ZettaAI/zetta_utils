
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
            expand_bbox_resolution=self.expand_bbox_resolution,
            expand_bbox_backend=self.expand_bbox_backend,
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