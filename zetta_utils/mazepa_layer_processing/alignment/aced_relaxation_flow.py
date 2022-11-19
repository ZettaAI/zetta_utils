from __future__ import annotations

from typing import Literal, Optional

from zetta_utils import alignment, builder, mazepa
from zetta_utils.bcube import BoundingCube
from zetta_utils.layer.volumetric import (
    VolumetricIndex,
    VolumetricIndexChunker,
    VolumetricLayer,
)
from zetta_utils.typing import IntVec3D, Vec3D

from ..common import build_chunked_volumetric_callable_flow_type


@builder.register("build_aced_relaxation_flow")
def build_aced_relaxation_flow(
    chunk_size: IntVec3D,
    bcube: BoundingCube,
    dst_resolution: Vec3D,
    dst: VolumetricLayer,
    field: VolumetricLayer,
    crop: IntVec3D,
    num_iter: int = 100,
    lr: float = 0.3,
    rigidity_weight: float = 10.0,
    fix: Optional[Literal["first", "last", "both"]] = "first",
) -> mazepa.Flow:
    flow_type = build_chunked_volumetric_callable_flow_type(
        fn=alignment.aced_relaxation.perform_aced_relaxation,
        chunker=VolumetricIndexChunker(chunk_size=chunk_size),
        crop=crop,
    )
    flow = flow_type(
        idx=VolumetricIndex(bcube=bcube, resolution=dst_resolution),
        dst=dst,
        field=field,
        num_iter=num_iter,
        lr=lr,
        rigidity_weight=rigidity_weight,
        fix=fix,
    )
    return flow
