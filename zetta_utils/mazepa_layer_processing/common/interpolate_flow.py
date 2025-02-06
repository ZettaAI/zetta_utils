from __future__ import annotations

from typing import Sequence, Union, cast

from numpy import typing as npt

from zetta_utils import builder, mazepa, tensor_ops
from zetta_utils.common import ComparablePartial
from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric.protocols import VolumetricBasedLayerProtocol
from zetta_utils.mazepa.flows import sequential_flow
from zetta_utils.mazepa_layer_processing.common.subchunkable_apply_flow import (
    build_subchunkable_apply_flow,
)

from . import VolumetricCallableOperation


def _interpolate(
    src: npt.NDArray,
    scale_factor: Union[float, Sequence[float]],
    mode: tensor_ops.InterpolationMode,
    mask_value_thr: float = 0,
) -> npt.NDArray:
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


@builder.register("build_interpolate_flow")
def build_interpolate_flow(  # pylint: disable=too-many-locals
    src: VolumetricBasedLayerProtocol,
    dst: VolumetricBasedLayerProtocol | None,
    src_resolution: Sequence[float],
    dst_resolutions: Sequence[Sequence[float]] | Sequence[float],
    mode: tensor_ops.InterpolationMode,
    processing_chunk_sizes: Sequence[Sequence[int]],
    level_intermediaries_dirs: Sequence[str | None] | None = None,
    skip_intermediaries: bool = False,
    expand_bbox_resolution: bool = False,
    expand_bbox_backend: bool = False,
    expand_bbox_processing: bool = True,
    shrink_processing_chunk: bool = False,
    auto_divisibility: bool = False,
    bbox: BBox3D | None = None,
    auto_bbox: bool = False,
    dst_tighten_bounds: bool = False,
) -> mazepa.Flow:
    if dst is None:
        dst = src

    if isinstance(dst_resolutions[0], float):
        dst_resolutions_list = cast(Sequence[Sequence[float]], [dst_resolutions])
    else:
        dst_resolutions_list = cast(Sequence[Sequence[float]], dst_resolutions)

    dst_resolutions_vec = [Vec3D(*e) for e in dst_resolutions_list]

    dst_resolutions_vec_sorted = sorted(dst_resolutions_vec, key=tuple)

    for i in range(len(dst_resolutions_vec_sorted) - 1):
        a = dst_resolutions_vec_sorted[i]
        b = dst_resolutions_vec_sorted[i + 1]
        if not a <= b:
            raise RuntimeError(
                "Cannot find a strictly increasing order for the given resolutions: " f"{a} {b}"
            )

    stages = []
    last_res = Vec3D(*src_resolution)
    last_src = src
    for i, dst_res in enumerate(dst_resolutions_vec_sorted):
        stages.append(
            build_subchunkable_apply_flow(
                op=make_interpolate_operation(
                    res_change_mult=dst_res / last_res,
                    mode=mode,
                ),
                dst=dst,
                dst_resolution=dst_res,
                processing_chunk_sizes=processing_chunk_sizes,
                level_intermediaries_dirs=level_intermediaries_dirs,
                skip_intermediaries=skip_intermediaries,
                expand_bbox_resolution=expand_bbox_resolution,
                expand_bbox_backend=expand_bbox_backend,
                expand_bbox_processing=expand_bbox_processing,
                shrink_processing_chunk=shrink_processing_chunk,
                auto_divisibility=auto_divisibility,
                bbox=bbox,
                auto_bbox=auto_bbox,
                dst_tighten_bounds=dst_tighten_bounds,
                op_kwargs={"src": last_src},
            )
        )
        last_res = dst_res
        last_src = dst
    result = sequential_flow(stages=stages)
    return result
