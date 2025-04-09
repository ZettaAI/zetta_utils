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
    max_reduction_chunk_size: Sequence[int] | None = None
    expand_bbox_resolution: bool = False
    expand_bbox_backend: bool = False
    expand_bbox_processing: bool = False
    shrink_processing_chunk: bool = False

    operation: ComputeFieldOperation = attrs.field(init=False)

    res_change_mult: Sequence[float] = (1, 1, 1)
    translation_granularity: int = 1

    src: VolumetricLayer | None = None
    tgt: VolumetricLayer | None = None

    src_rig_weight: VolumetricLayer | None = None
    tgt_rig_weight: VolumetricLayer | None = None
    src_mse_weight: VolumetricLayer | None = None
    tgt_mse_weight: VolumetricLayer | None = None
    src_pinned_field: VolumetricLayer | None = None

    def __attrs_post_init__(self):
        self.operation = ComputeFieldOperation(
            fn=self.fn,
            res_change_mult=self.res_change_mult,
            translation_granularity=self.translation_granularity,
        )

    @property
    def input_resolution(self) -> Vec3D[float]:
        return self.operation.get_input_resolution(Vec3D(*self.dst_resolution))


def _set_up_offsets(  # pylint: disable=too-many-branches
    stages: list[ComputeFieldStage],
    src_field: VolumetricLayer | None = None,
    tgt_field: VolumetricLayer | None = None,
    src: VolumetricLayer | None = None,
    tgt: VolumetricLayer | None = None,
    tgt_offset: Sequence[float] = (0, 0, 0),
    src_offset: Sequence[float] = (0, 0, 0),
    offset_resolution: Sequence[float] | None = None,
    src_rig_weight: VolumetricLayer | None = None,
    tgt_rig_weight: VolumetricLayer | None = None,
    src_mse_weight: VolumetricLayer | None = None,
    tgt_mse_weight: VolumetricLayer | None = None,
    src_pinned_field: VolumetricLayer | None = None,
) -> tuple[
    list[ComputeFieldStage],
    VolumetricLayer | None,
    VolumetricLayer | None,
    VolumetricLayer | None,
    VolumetricLayer | None,
    VolumetricLayer | None,
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

        for stage in stages:
            if stage.src is not None:
                stage.src = stage.src.with_procs(
                    index_procs=(src_offsetter,) + stage.src.index_procs
                )
            if stage.tgt is not None:
                stage.tgt = stage.tgt.with_procs(
                    index_procs=(tgt_offsetter,) + stage.tgt.index_procs
                )
            if stage.src_rig_weight is not None:
                stage.src_rig_weight = stage.src_rig_weight.with_procs(
                    index_procs=(src_offsetter,) + stage.src_rig_weight.index_procs
                )
            if stage.tgt_rig_weight is not None:
                stage.tgt_rig_weight = stage.tgt_rig_weight.with_procs(
                    index_procs=(tgt_offsetter,) + stage.tgt_rig_weight.index_procs
                )
            if stage.src_mse_weight is not None:
                stage.src_mse_weight = stage.src_mse_weight.with_procs(
                    index_procs=(src_offsetter,) + stage.src_mse_weight.index_procs
                )
            if stage.tgt_mse_weight is not None:
                stage.tgt_mse_weight = stage.tgt_mse_weight.with_procs(
                    index_procs=(tgt_offsetter,) + stage.tgt_mse_weight.index_procs
                )
            if stage.src_pinned_field is not None:
                stage.src_pinned_field = stage.src_pinned_field.with_procs(
                    index_procs=(src_offsetter,) + stage.src_pinned_field.index_procs
                )

    return (
        stages,
        src,
        tgt,
        src_field,
        tgt_field,
        src_rig_weight,
        tgt_rig_weight,
        src_mse_weight,
        tgt_mse_weight,
        src_pinned_field,
    )


@builder.register("ComputeFieldMultistageFlowSchema")
@mazepa.flow_schema_cls
@attrs.mutable
class ComputeFieldMultistageFlowSchema:
    stages: list[ComputeFieldStage]
    tmp_layer_dir: str
    tmp_layer_factory: Callable[..., VolumetricLayer]

    def flow(  # pylint: disable=too-many-locals, too-many-branches
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
        src_rig_weight: VolumetricLayer | None = None,
        tgt_rig_weight: VolumetricLayer | None = None,
        src_mse_weight: VolumetricLayer | None = None,
        tgt_mse_weight: VolumetricLayer | None = None,
        src_pinned_field: VolumetricLayer | None = None,
    ):
        (
            stages,
            src,
            tgt,
            src_field,
            tgt_field,
            src_rig_weight,
            tgt_rig_weight,
            src_mse_weight,
            tgt_mse_weight,
            src_pinned_field,
        ) = _set_up_offsets(
            self.stages,
            src_field=src_field,
            tgt_field=tgt_field,
            src=src,
            tgt=tgt,
            tgt_offset=tgt_offset,
            src_offset=src_offset,
            offset_resolution=offset_resolution,
            src_rig_weight=src_rig_weight,
            tgt_rig_weight=tgt_rig_weight,
            src_mse_weight=src_mse_weight,
            tgt_mse_weight=tgt_mse_weight,
            src_pinned_field=src_pinned_field,
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

            stage_src_rig_weight = stage.src_rig_weight or src_rig_weight
            stage_tgt_rig_weight = stage.tgt_rig_weight or tgt_rig_weight
            stage_src_mse_weight = stage.src_mse_weight or src_mse_weight
            stage_tgt_mse_weight = stage.tgt_mse_weight or tgt_mse_weight
            stage_src_pinned_field = stage.src_pinned_field or src_pinned_field
            stage_src_field = prev_dst or src_field

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
                max_reduction_chunk_size=stage.max_reduction_chunk_size,
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
                src_rig_weight=stage_src_rig_weight,
                tgt_rig_weight=stage_tgt_rig_weight,
                src_mse_weight=stage_src_mse_weight,
                tgt_mse_weight=stage_tgt_mse_weight,
                src_pinned_field=stage_src_pinned_field,
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
    src_rig_weight: VolumetricLayer | None = None,
    tgt_rig_weight: VolumetricLayer | None = None,
    src_mse_weight: VolumetricLayer | None = None,
    tgt_mse_weight: VolumetricLayer | None = None,
    src_pinned_field: VolumetricLayer | None = None,
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
        src_rig_weight=src_rig_weight,
        tgt_rig_weight=tgt_rig_weight,
        src_mse_weight=src_mse_weight,
        tgt_mse_weight=tgt_mse_weight,
        src_pinned_field=src_pinned_field,
    )
    return flow
