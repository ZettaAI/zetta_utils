from __future__ import annotations

import torch

from zetta_utils import builder, mazepa
from zetta_utils.geometry import BBox3D, IntVec3D, Vec3D
from zetta_utils.layer.volumetric import (
    VolumetricIndex,
    VolumetricIndexChunker,
    VolumetricLayer,
)

from . import build_chunked_volumetric_callable_flow_schema


def _apply_mask(
    src: torch.Tensor,
    mask: torch.Tensor,
    fill_value: float = 0,
) -> torch.Tensor:
    result = src
    result[mask > 0] = fill_value
    result = result.to(src.dtype)
    return result


@builder.register(
    "build_apply_mask_flow", cast_to_vec3d=["dst_resolution"], cast_to_intvec3d=["chunk_size"]
)
def build_apply_mask_flow(
    chunk_size: IntVec3D,
    bbox: BBox3D,
    dst_resolution: Vec3D,
    src: VolumetricLayer,
    dst: VolumetricLayer,
    mask: VolumetricLayer,
    fill_value: float = 0.0,
) -> mazepa.Flow:
    flow_schema = build_chunked_volumetric_callable_flow_schema(
        fn=_apply_mask,
        chunker=VolumetricIndexChunker(chunk_size=chunk_size),
    )
    flow = flow_schema(
        idx=VolumetricIndex(bbox=bbox, resolution=dst_resolution),
        dst=dst,
        src=src,
        mask=mask,
        fill_value=fill_value,
    )
    return flow
