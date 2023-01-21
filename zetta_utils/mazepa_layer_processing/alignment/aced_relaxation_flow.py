from __future__ import annotations

from typing import Literal, Optional

from zetta_utils import alignment, builder, mazepa
from zetta_utils.bbox import BBox3D
from zetta_utils.layer.volumetric import (
    VolumetricIndex,
    VolumetricIndexChunker,
    VolumetricLayer,
)
from zetta_utils.typing import IntVec3D, Vec3D

from ..common import build_chunked_volumetric_callable_flow_schema


@builder.register(
    "build_aced_relaxation_flow",
    cast_to_vec3d=["dst_resolution"],
    cast_to_intvec3d=["chunk_size", "crop_pad"],
)
def build_aced_relaxation_flow(
    chunk_size: IntVec3D,
    bbox: BBox3D,
    dst_resolution: Vec3D,
    dst: VolumetricLayer,
    field: VolumetricLayer,
    crop_pad: IntVec3D,
    num_iter: int = 100,
    lr: float = 0.3,
    rigidity_weight: float = 10.0,
    fix: Optional[Literal["first", "last", "both"]] = "first",
) -> mazepa.Flow:
    flow_schema = build_chunked_volumetric_callable_flow_schema(
        fn=alignment.aced_relaxation.perform_aced_relaxation,
        chunker=VolumetricIndexChunker(chunk_size=chunk_size),
        crop_pad=crop_pad,
    )
    flow = flow_schema(
        idx=VolumetricIndex(bbox=bbox, resolution=dst_resolution),
        dst=dst,
        field=field,
        num_iter=num_iter,
        lr=lr,
        rigidity_weight=rigidity_weight,
        fix=fix,
    )
    return flow
