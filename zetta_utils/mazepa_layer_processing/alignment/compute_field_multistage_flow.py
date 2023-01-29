from __future__ import annotations

import copy
import os
from typing import Callable

import attrs

from zetta_utils import builder, mazepa
from zetta_utils.geometry import BBox3D, IntVec3D, Vec3D
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

    chunk_size: IntVec3D
    dst_resolution: Vec3D

    operation: ComputeFieldOperation = attrs.field(init=False)

    crop_pad: int = 0
    res_change_mult: Vec3D = Vec3D(1, 1, 1)

    src: VolumetricLayer | None = None
    tgt: VolumetricLayer | None = None

    def __attrs_post_init__(self):
        self.operation = ComputeFieldOperation(
            fn=self.fn,
            crop_pad=self.crop_pad,
            res_change_mult=self.res_change_mult,
        )

    @property
    def input_resolution(self):
        return self.operation.get_input_resolution(self.dst_resolution)


def _set_up_offsets(
    stages: list[ComputeFieldStage],
    src_field: VolumetricLayer | None = None,
    tgt_field: VolumetricLayer | None = None,
    src: VolumetricLayer | None = None,
    tgt: VolumetricLayer | None = None,
    tgt_offset: Vec3D = Vec3D(0, 0, 0),
    src_offset: Vec3D = Vec3D(0, 0, 0),
    offset_resolution: Vec3D | None = None,
) -> tuple[
    list[ComputeFieldStage],
    VolumetricLayer | None,
    VolumetricLayer | None,
    VolumetricLayer | None,
    VolumetricLayer | None,
]:
    if tgt_offset != Vec3D(0, 0, 0) or src_offset != Vec3D(0, 0, 0):
        stages = copy.deepcopy(stages)
        if offset_resolution is None:
            raise Exception(
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
        tgt_offset: Vec3D = Vec3D(0, 0, 0),
        src_offset: Vec3D = Vec3D(0, 0, 0),
        offset_resolution: Vec3D | None = None,
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
                chunk_size=stage.chunk_size,
                operation=stage.operation,
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
    tgt_offset: Vec3D = Vec3D(0, 0, 0),
    src_offset: Vec3D = Vec3D(0, 0, 0),
    offset_resolution: Vec3D | None = None,
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
