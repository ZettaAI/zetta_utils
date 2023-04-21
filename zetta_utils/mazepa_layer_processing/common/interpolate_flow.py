from __future__ import annotations

from typing import Optional, Sequence, Union

import torch

from zetta_utils import builder, mazepa, tensor_ops
from zetta_utils.common import ComparablePartial
from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric import (
    VolumetricIndex,
    VolumetricIndexChunker,
    VolumetricLayer,
)

from . import VolumetricCallableOperation, build_chunked_volumetric_callable_flow_schema


def _interpolate(
    src: torch.Tensor,
    scale_factor: Union[float, Sequence[float]],
    mode: tensor_ops.InterpolationMode,
    mask_value_thr: float = 0,
) -> torch.Tensor:
    # This dummy function is necessary to rename `src` to `data` arg
    result = tensor_ops.interpolate(
        data=src,
        scale_factor=scale_factor,
        mode=mode,
        mask_value_thr=mask_value_thr,
        unsqueeze_input_to=5,
    )
    return result


@builder.register("InterpolateOperation")
def make_interpolate_operation(
    res_change_mult: Sequence[float],
    mode: tensor_ops.InterpolationMode,
    mask_value_thr: float = 0,
):
    op = VolumetricCallableOperation(
        fn=ComparablePartial(
            _interpolate,
            mode=mode,
            scale_factor=1 / Vec3D(*res_change_mult),
            mask_value_thr=mask_value_thr,
        ),
        res_change_mult=Vec3D(*res_change_mult),
        operation_name=f"Interpolate<{mode}>",
    )
    return op


# TODO: remove as soon as subchunkable can support `res_change_mult`
@builder.register("build_interpolate_flow")
def build_interpolate_flow(
    chunk_size: Sequence[int],
    bbox: BBox3D,
    src_resolution: Sequence[float],
    dst_resolution: Sequence[float],
    src: VolumetricLayer,
    mode: tensor_ops.InterpolationMode,
    dst: Optional[VolumetricLayer] = None,
    mask_value_thr: float = 0,
) -> mazepa.Flow:

    if dst is None:
        dst = src

    scale_factor = Vec3D(*src_resolution) / Vec3D(*dst_resolution)
    res_change_mult = Vec3D(*dst_resolution) / Vec3D(*src_resolution)

    flow_schema = build_chunked_volumetric_callable_flow_schema(
        fn=_interpolate,
        res_change_mult=res_change_mult,
        chunker=VolumetricIndexChunker(chunk_size=Vec3D[int](*chunk_size)),
    )
    flow = flow_schema(
        idx=VolumetricIndex(bbox=bbox, resolution=Vec3D(*dst_resolution)),
        dst=dst,
        src=src,
        scale_factor=scale_factor,
        mode=mode,
        mask_value_thr=mask_value_thr,
    )
    return flow
