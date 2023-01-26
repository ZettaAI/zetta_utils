from __future__ import annotations

from typing import Literal, Optional

from zetta_utils import alignment, builder, mazepa
from zetta_utils.geometry import BBox3D, IntVec3D, Vec3D
from zetta_utils.layer.volumetric import (
    VolumetricIndex,
    VolumetricIndexChunker,
    VolumetricLayer,
)

from ..common import build_chunked_volumetric_callable_flow_schema


@builder.register("build_get_match_offsets_flow")
def build_get_match_offsets_flow(
    chunk_size: IntVec3D,
    bbox: BBox3D,
    dst_resolution: Vec3D,
    non_tissue: VolumetricLayer,
    dst: VolumetricLayer,
    misd_mask_zm1: VolumetricLayer,
    misd_mask_zm2: Optional[VolumetricLayer] = None,
    misd_mask_zm3: Optional[VolumetricLayer] = None,
) -> mazepa.Flow:
    flow_schema = build_chunked_volumetric_callable_flow_schema(
        fn=alignment.aced_relaxation.get_aced_match_offsets,
        chunker=VolumetricIndexChunker(chunk_size=chunk_size),
    )
    flow = flow_schema(
        idx=VolumetricIndex(bbox=bbox, resolution=dst_resolution),
        non_tissue=non_tissue,
        dst=dst,
        misd_mask_zm1=misd_mask_zm1,
        misd_mask_zm2=misd_mask_zm2,
        misd_mask_zm3=misd_mask_zm3,
    )
    return flow


@builder.register("build_aced_relaxation_flow")
def build_aced_relaxation_flow(
    chunk_size: IntVec3D,
    bbox: BBox3D,
    dst_resolution: Vec3D,
    dst: VolumetricLayer,
    match_offsets: VolumetricLayer,
    field_zm1: VolumetricLayer,
    crop_pad: IntVec3D,
    field_zm2: Optional[VolumetricLayer] = None,
    field_zm3: Optional[VolumetricLayer] = None,
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
        match_offsets=match_offsets,
        field_zm1=field_zm1,
        field_zm2=field_zm2,
        field_zm3=field_zm3,
        num_iter=num_iter,
        lr=lr,
        rigidity_weight=rigidity_weight,
        fix=fix,
    )
    return flow
