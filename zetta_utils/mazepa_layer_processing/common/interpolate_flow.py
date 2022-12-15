from __future__ import annotations

from typing import Optional, Sequence, Union

import torch

from zetta_utils import builder, mazepa, tensor_ops
from zetta_utils.bcube import BoundingCube
from zetta_utils.layer.volumetric import (
    VolumetricIndex,
    VolumetricIndexChunker,
    VolumetricLayer,
)
from zetta_utils.typing import IntVec3D, Vec3D

from . import build_chunked_volumetric_callable_flow_schema


def _interpolate(
    src: torch.Tensor,
    scale_factor: Union[float, Sequence[float]],
    mode: tensor_ops.InterpolationMode,
    mask_value_thr: float = 0,
) -> torch.Tensor:
    return tensor_ops.interpolate(
        data=src,
        scale_factor=scale_factor,
        mode=mode,
        mask_value_thr=mask_value_thr,
        unsqueeze_input_to=5,
    )


@builder.register(
    "build_interpolate_flow",
    cast_to_vec3d=["src_resolution", "dst_resolution"],
    cast_to_intvec3d=["chunk_size"],
)
def build_interpolate_flow(
    chunk_size: IntVec3D,
    bcube: BoundingCube,
    src_resolution: Vec3D,
    dst_resolution: Vec3D,
    src: VolumetricLayer,
    mode: tensor_ops.InterpolationMode,
    dst: Optional[VolumetricLayer] = None,
    mask_value_thr: float = 0,
) -> mazepa.Flow:

    if dst is None:
        dst = src

    # must use generator since np.float64 is not a float
    scale_factor = src_resolution / dst_resolution
    reveal_type(scale_factor)
    #    res_change_mult = tuple(float(e) for e in dst_resolution / src_resolution)
    res_change_mult = dst_resolution / src_resolution

    flow_schema = build_chunked_volumetric_callable_flow_schema(
        fn=_interpolate,
        res_change_mult=res_change_mult,
        chunker=VolumetricIndexChunker(chunk_size=chunk_size),
    )
    flow = flow_schema(
        idx=VolumetricIndex(bcube=bcube, resolution=dst_resolution),
        dst=dst,
        src=src,
        scale_factor=scale_factor,
        mode=mode,
        mask_value_thr=mask_value_thr,
    )
    return flow
