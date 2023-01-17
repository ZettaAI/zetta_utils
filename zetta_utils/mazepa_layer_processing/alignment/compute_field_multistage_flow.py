import copy
import os
from typing import Callable, List, Optional

import attrs

from zetta_utils import builder, mazepa
from zetta_utils.bcube import BoundingCube
from zetta_utils.layer.volumetric import VolumetricIndexTranslator, VolumetricLayer
from zetta_utils.mazepa_layer_processing.common import build_interpolate_flow
from zetta_utils.typing import IntVec3D, Vec3D

from .compute_field_flow import ComputeFieldFlowSchema, ComputeFieldOperation


@builder.register(
    "ComputeFieldStage", cast_to_vec3d=["dst_resolution"], cast_to_intvec3d=["chunk_size"]
)
@attrs.mutable
class ComputeFieldStage:
    dst_resolution: Vec3D
    operation: ComputeFieldOperation
    chunk_size: IntVec3D

    crop: int = 0

    src: Optional[VolumetricLayer] = None
    tgt: Optional[VolumetricLayer] = None

    @property
    def input_resolution(self) -> Vec3D:
        return self.operation.get_input_resolution(self.dst_resolution)


@builder.register(
    "ComputeFieldMultistageFlowSchema",
    cast_to_vec3d=["tgt_offset", "src_offset", "offset_resolution"],
)
@mazepa.flow_schema_cls
@attrs.mutable
class ComputeFieldMultistageFlowSchema:
    stages: List[ComputeFieldStage]
    tmp_layer_dir: str
    tmp_layer_factory: Callable[..., VolumetricLayer]

    def flow(  # pylint: disable=too-many-locals
        self,
        bcube: BoundingCube,
        dst: VolumetricLayer,
        src_field: Optional[VolumetricLayer] = None,
        tgt_field: Optional[VolumetricLayer] = None,
        src: Optional[VolumetricLayer] = None,
        tgt: Optional[VolumetricLayer] = None,
        tgt_offset: Vec3D = Vec3D(0, 0, 0),
        src_offset: Vec3D = Vec3D(0, 0, 0),
        offset_resolution: Optional[Vec3D] = None,
    ):
        if tgt_offset != Vec3D(0, 0, 0) or src_offset != Vec3D(0, 0, 0):
            if offset_resolution is None:
                raise Exception(
                    "Must provide `offset_resolution` when either `src_offset` or `tgt_offset` "
                    "are given."
                )
            src_offset_in_unit = src_offset * offset_resolution
            tgt_offset_in_unit = tgt_offset * offset_resolution
        else:
            src_offset_in_unit = Vec3D(0, 0, 0)
            tgt_offset_in_unit = Vec3D(0, 0, 0)

        if tgt_field is not None:
            tgt_field = copy.deepcopy(tgt_field)
            tgt_field.index_adjs.insert(0, VolumetricIndexTranslator(offset=tgt_offset_in_unit))

        if src_field is not None:
            src_field = copy.deepcopy(src_field)
            src_field.index_adjs.insert(0, VolumetricIndexTranslator(offset=src_offset_in_unit))

        prev_dst: Optional[VolumetricLayer] = None

        for i, stage in enumerate(self.stages):
            if i > 0 and stage.input_resolution != self.stages[i - 1].dst_resolution:
                assert prev_dst is not None

                yield build_interpolate_flow(
                    chunk_size=stage.chunk_size,
                    bcube=bcube,
                    src=prev_dst,
                    src_resolution=self.stages[i - 1].dst_resolution,
                    dst_resolution=stage.input_resolution,
                    mode="field",
                )
                yield mazepa.Dependency()

            if i == len(self.stages) - 1:
                stage_dst = dst
            else:
                stage_dst_path = os.path.join(self.tmp_layer_dir, f"stage_{i}")
                stage_dst = self.tmp_layer_factory(path=stage_dst_path)

            if stage.src is not None:
                stage_src = copy.deepcopy(stage.src)
            else:
                assert src is not None
                stage_src = copy.deepcopy(src)

            if stage.tgt is not None:
                stage_tgt = copy.deepcopy(stage.tgt)
            else:
                assert tgt is not None
                stage_tgt = copy.deepcopy(tgt)

            if prev_dst is None:
                stage_src_field = src_field
            else:
                stage_src_field = prev_dst

            stage_tgt.index_adjs.insert(0, VolumetricIndexTranslator(offset=tgt_offset_in_unit))

            stage_src.index_adjs.insert(0, VolumetricIndexTranslator(offset=src_offset_in_unit))

            stage_cf_flow_schema = ComputeFieldFlowSchema(
                chunk_size=stage.chunk_size,
                operation=stage.operation,
            )
            yield stage_cf_flow_schema(
                bcube=bcube,
                dst_resolution=stage.dst_resolution,
                dst=stage_dst,
                src=stage_src,
                tgt=stage_tgt,
                src_field=stage_src_field,
                tgt_field=tgt_field,
            )
            yield mazepa.Dependency()

            prev_dst = stage_dst


@builder.register(
    "build_compute_field_multistage_flow",
    cast_to_vec3d=["tgt_offset", "src_offset", "offset_resolution"],
)
def build_compute_field_multistage_flow(
    stages: List[ComputeFieldStage],
    tmp_layer_dir: str,
    tmp_layer_factory: Callable[..., VolumetricLayer],
    bcube: BoundingCube,
    dst: VolumetricLayer,
    src_field: Optional[VolumetricLayer] = None,
    tgt_field: Optional[VolumetricLayer] = None,
    src: Optional[VolumetricLayer] = None,
    tgt: Optional[VolumetricLayer] = None,
    tgt_offset: Vec3D = Vec3D(0, 0, 0),
    src_offset: Vec3D = Vec3D(0, 0, 0),
    offset_resolution: Optional[Vec3D] = None,
) -> mazepa.Flow:
    flow_schema = ComputeFieldMultistageFlowSchema(
        stages=stages,
        tmp_layer_dir=tmp_layer_dir,
        tmp_layer_factory=tmp_layer_factory,
    )
    flow = flow_schema(
        bcube=bcube,
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
