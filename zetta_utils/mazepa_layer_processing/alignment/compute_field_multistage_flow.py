from __future__ import annotations

import copy
import os
from typing import Callable, Literal, Sequence, Union

import attrs

from zetta_utils import builder, mazepa
from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric import (
    DataResolutionInterpolator,
    VolumetricIndexTranslator,
    VolumetricLayer,
)

from .compute_field_flow import (
    ComputeFieldFlowSchema,
    ComputeFieldFn,
    ComputeFieldOperation,
)


@builder.register("ComputeFieldStage")
@attrs.mutable
class ComputeFieldStage:
    fn: ComputeFieldFn

    dst_resolution: Sequence[float]

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

    operation: ComputeFieldOperation = attrs.field(init=False)

    res_change_mult: Sequence[float] = (1, 1, 1)

    src: VolumetricLayer | None = None
    tgt: VolumetricLayer | None = None

    def __attrs_post_init__(self):
        self.operation = ComputeFieldOperation(
            fn=self.fn,
            res_change_mult=self.res_change_mult,
        )

    @property
    def input_resolution(self) -> Vec3D[float]:
        return self.operation.get_input_resolution(Vec3D(*self.dst_resolution))


def _set_up_offsets(
    stages: list[ComputeFieldStage],
    src_field: VolumetricLayer | None = None,
    tgt_field: VolumetricLayer | None = None,
    src: VolumetricLayer | None = None,
    tgt: VolumetricLayer | None = None,
    tgt_offset: Sequence[float] = (0, 0, 0),
    src_offset: Sequence[float] = (0, 0, 0),
    offset_resolution: Sequence[float] | None = None,
) -> tuple[
    list[ComputeFieldStage],
    VolumetricLayer | None,
    VolumetricLayer | None,
    VolumetricLayer | None,
    VolumetricLayer | None,
]:
    if not all(e == 0 for e in tgt_offset) or not all(e == 0 for e in src_offset):
        stages = copy.deepcopy(stages)
        if offset_resolution is None:
            raise ValueError(
                "Must provide `offset_resolution` when either `src_offset` or `tgt_offset` "
                "are given."
            )
        src_offsetter = VolumetricIndexTranslator(src_offset, offset_resolution)
        tgt_offsetter = VolumetricIndexTranslator(tgt_offset, offset_resolution)

        if src is not None:
            src = src.with_procs(index_procs=(src_offsetter,) + src.index_procs)

        if tgt is not None:
            tgt = tgt.with_procs(index_procs=(tgt_offsetter,) + tgt.index_procs)

        if src_field is not None:
            src_field = src_field.with_procs(index_procs=(src_offsetter,) + src_field.index_procs)

        if tgt_field is not None:
            tgt_field = tgt_field.with_procs(index_procs=(tgt_offsetter,) + tgt_field.index_procs)

        for stage in stages:
            if stage.src is not None:
                stage.src = stage.src.with_procs(
                    index_procs=(src_offsetter,) + stage.src.index_procs
                )
            if stage.tgt is not None:
                stage.tgt = stage.tgt.with_procs(
                    index_procs=(tgt_offsetter,) + stage.tgt.index_procs
                )
    return stages, src, tgt, src_field, tgt_field


@builder.register("ComputeFieldMultistageFlowSchema")
@mazepa.flow_schema_cls
@attrs.mutable
class ComputeFieldMultistageFlowSchema:
    stages: list[ComputeFieldStage]
    tmp_layer_dir: str
    tmp_layer_factory: Callable[..., VolumetricLayer]

    def flow(  # pylint: disable=too-many-locals
        self,
        bbox: BBox3D,
        dst: VolumetricLayer,
        src_field: VolumetricLayer | None = None,
        tgt_field: VolumetricLayer | None = None,
        src: VolumetricLayer | None = None,
        tgt: VolumetricLayer | None = None,
        tgt_offset: Sequence[float] = (0, 0, 0),
        src_offset: Sequence[float] = (0, 0, 0),
        offset_resolution: Sequence[float] | None = None,
    ):
        stages, src, tgt, src_field, tgt_field = _set_up_offsets(
            self.stages,
            src_field=src_field,
            tgt_field=tgt_field,
            src=src,
            tgt=tgt,
            tgt_offset=tgt_offset,
            src_offset=src_offset,
            offset_resolution=offset_resolution,
        )

        prev_dst: VolumetricLayer | None = None

        for i, stage in enumerate(stages):
            if i > 0 and stage.input_resolution != stages[i - 1].dst_resolution:
                assert prev_dst is not None
                prev_dst = prev_dst.with_procs(
                    read_procs=(
                        DataResolutionInterpolator(
                            data_resolution=stages[i - 1].dst_resolution,
                            interpolation_mode="field",
                        ),
                    )
                )

            if i == len(stages) - 1:
                stage_dst = dst
            else:
                stage_dst_path = os.path.join(self.tmp_layer_dir, f"stage_{i}")
                stage_dst = self.tmp_layer_factory(path=stage_dst_path)

            if stage.src is not None:
                stage_src = stage.src
            else:
                assert src is not None
                stage_src = src

            if stage.tgt is not None:
                stage_tgt = stage.tgt
            else:
                assert tgt is not None
                stage_tgt = tgt

            if prev_dst is None:
                stage_src_field = src_field
            else:
                stage_src_field = prev_dst

            stage_cf_flow_schema = ComputeFieldFlowSchema(
                operation=stage.operation,
                processing_chunk_sizes=stage.processing_chunk_sizes,
                processing_crop_pads=stage.processing_crop_pads,
                processing_blend_pads=stage.processing_blend_pads,
                processing_blend_modes=stage.processing_blend_modes,
                expand_bbox_resolution=stage.expand_bbox_resolution,
                expand_bbox_backend=stage.expand_bbox_backend,
                expand_bbox_processing=stage.expand_bbox_processing,
                shrink_processing_chunk=stage.shrink_processing_chunk,
                max_reduction_chunk_sizes=stage.max_reduction_chunk_sizes,
                level_intermediaries_dirs=stage.level_intermediaries_dirs,
            )
            yield stage_cf_flow_schema(
                bbox=bbox,
                dst_resolution=stage.dst_resolution,
                dst=stage_dst,
                src=stage_src,
                tgt=stage_tgt,
                src_field=stage_src_field,
                tgt_field=tgt_field,
            )
            yield mazepa.Dependency()

            prev_dst = stage_dst


@builder.register("build_compute_field_multistage_flow")
def build_compute_field_multistage_flow(
    stages: list[ComputeFieldStage],
    tmp_layer_dir: str,
    tmp_layer_factory: Callable[..., VolumetricLayer],
    bbox: BBox3D,
    dst: VolumetricLayer,
    src_field: VolumetricLayer | None = None,
    tgt_field: VolumetricLayer | None = None,
    src: VolumetricLayer | None = None,
    tgt: VolumetricLayer | None = None,
    tgt_offset: Sequence[float] = (0, 0, 0),
    src_offset: Sequence[float] = (0, 0, 0),
    offset_resolution: Sequence[float] | None = None,
) -> mazepa.Flow:
    flow_schema = ComputeFieldMultistageFlowSchema(
        stages=stages,
        tmp_layer_dir=tmp_layer_dir,
        tmp_layer_factory=tmp_layer_factory,
    )
    flow = flow_schema(
        bbox=bbox,
        dst=dst,
        src_field=src_field,
        tgt_field=tgt_field,
        src=src,
        tgt=tgt,
        tgt_offset=tgt_offset,
        src_offset=src_offset,
        offset_resolution=offset_resolution,
    )
    return flow
