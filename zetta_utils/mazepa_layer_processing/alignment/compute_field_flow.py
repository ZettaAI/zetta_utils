import copy
from typing import Optional

import attrs

from zetta_utils import builder, mazepa
from zetta_utils.geometry import BBox3D, IntVec3D, Vec3D
from zetta_utils.layer.volumetric import (
    VolumetricIndex,
    VolumetricIndexChunker,
    VolumetricIndexTranslator,
    VolumetricLayer,
)
from zetta_utils.mazepa_layer_processing.common import build_chunked_apply_flow

from ..operation_protocols import ComputeFieldOpProtocol


@builder.register("ComputeFieldFlowSchema")
@mazepa.flow_schema_cls
@attrs.mutable
class ComputeFieldFlowSchema:
    chunk_size: IntVec3D
    operation: ComputeFieldOpProtocol
    chunker: VolumetricIndexChunker = attrs.field(init=False)

    def __attrs_post_init__(self):
        self.chunker = VolumetricIndexChunker(chunk_size=self.chunk_size)

    def flow(
        self,
        bbox: BBox3D,
        dst_resolution: Vec3D,
        dst: VolumetricLayer,
        src: VolumetricLayer,
        tgt: Optional[VolumetricLayer] = None,
        src_field: Optional[VolumetricLayer] = None,
        tgt_offset: Vec3D = Vec3D(0, 0, 0),
    ):
        if tgt is None:
            tgt = src

        tgt = copy.deepcopy(tgt)
        input_resolution = self.operation.get_input_resolution(dst_resolution)
        tgt.index_procs.insert(
            0, VolumetricIndexTranslator(offset=tgt_offset, resolution=input_resolution)
        )
        cf_flow = build_chunked_apply_flow(
            operation=self.operation,  # type: ignore
            chunker=self.chunker,
            idx=VolumetricIndex(bbox=bbox, resolution=dst_resolution),
            dst=dst,  # type: ignore
            src=src,  # type: ignore
            tgt=tgt,  # type: ignore
            src_field=src_field,  # type: ignore
        )

        yield cf_flow


@builder.register("build_compute_field_flow")
def build_compute_field_flow(
    chunk_size: IntVec3D,
    operation: ComputeFieldOpProtocol,
    bbox: BBox3D,
    dst_resolution: Vec3D,
    dst: VolumetricLayer,
    src: VolumetricLayer,
    tgt: Optional[VolumetricLayer] = None,
    src_field: Optional[VolumetricLayer] = None,
    tgt_offset: Vec3D = Vec3D(0, 0, 0),
) -> mazepa.Flow:
    flow_schema = ComputeFieldFlowSchema(chunk_size=chunk_size, operation=operation)
    flow = flow_schema(
        bbox=bbox,
        dst_resolution=dst_resolution,
        dst=dst,
        src=src,
        tgt=tgt,
        src_field=src_field,
        tgt_offset=tgt_offset,
    )
    return flow
