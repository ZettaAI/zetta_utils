import os
from typing import Callable, List, Optional

import attrs

from zetta_utils import builder, mazepa
from zetta_utils.bcube import BoundingCube
from zetta_utils.layer.volumetric import VolumetricLayer
from zetta_utils.mazepa_layer_processing.common import build_interpolate_flow
from zetta_utils.typing import IntVec3D, Vec3D

from .compute_field_flow import ComputeFieldFlowType
from .compute_field_protocols import ComputeFieldOperation


@builder.register("ComputeFieldStage")
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


@builder.register("ComputeFieldMultistageFlowType")
@mazepa.flow_type_cls
@attrs.mutable
class ComputeFieldMultistageFlowType:
    stages: List[ComputeFieldStage]
    tmp_layer_dir: str
    tmp_layer_factory: Callable[..., VolumetricLayer]

    def flow(  # pylint: disable=too-many-locals
        self,
        bcube: BoundingCube,
        dst: VolumetricLayer,
        src_field: Optional[VolumetricLayer] = None,
        src: Optional[VolumetricLayer] = None,
        tgt: Optional[VolumetricLayer] = None,
        tgt_offset: Vec3D = (0, 0, 0),
    ):
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

            stage_cf_flow_type = ComputeFieldFlowType(
                chunk_size=stage.chunk_size,
                operation=stage.operation,
            )
            yield stage_cf_flow_type(
                bcube=bcube,
                dst_resolution=stage.dst_resolution,
                dst=stage_dst,
                src=stage_src,
                tgt=stage_tgt,
                src_field=stage_src_field,
                tgt_offset=tgt_offset,
            )
            yield mazepa.Dependency()

            prev_dst = stage_dst


@builder.register("build_compute_field_multistage_flow")
def build_compute_field_multistage_flow(
    stages: List[ComputeFieldStage],
    tmp_layer_dir: str,
    tmp_layer_factory: Callable[..., VolumetricLayer],
    bcube: BoundingCube,
    dst: VolumetricLayer,
    src_field: Optional[VolumetricLayer] = None,
    src: Optional[VolumetricLayer] = None,
    tgt: Optional[VolumetricLayer] = None,
    tgt_offset: Vec3D = (0, 0, 0),
) -> mazepa.Flow:
    flow_type = ComputeFieldMultistageFlowType(
        stages=stages,
        tmp_layer_dir=tmp_layer_dir,
        tmp_layer_factory=tmp_layer_factory,
    )
    flow = flow_type(
        bcube=bcube, dst=dst, src_field=src_field, src=src, tgt=tgt, tgt_offset=tgt_offset
    )
    return flow
