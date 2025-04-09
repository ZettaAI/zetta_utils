import math
from typing import Callable, Literal, Sequence

from sympy import divisors

from zetta_utils.geometry.bbox import BBox3D
from zetta_utils.geometry.vec import IntVec3D, Vec3D
from zetta_utils.layer.volumetric.layer import VolumetricLayer
from zetta_utils.log import get_logger
from zetta_utils.mazepa.flows import concurrent_flow
from zetta_utils.mazepa_layer_processing.common.subchunkable_apply_flow import (
    build_subchunkable_apply_flow,
)
from zetta_utils.mazepa_layer_processing.operation_protocols import VolumetricOpProtocol

logger = get_logger("zetta_utils")


def largest_divisor_leq(n: int, m: int) -> int:
    valid_divisors = [d for d in divisors(n) if d <= m]
    if not valid_divisors:
        raise ValueError(f"No divisor of {n} found that is <= {m}")
    return max(valid_divisors)


def get_chunk_size(
    chunk_size_base: Sequence[int], bbox_size: Sequence[int], must_be_divisible: bool
) -> Sequence[int]:
    chunk_size_clipped = [min(bbox_size[i], chunk_size_base[i]) for i in range(3)]
    chunk_size_final = chunk_size_clipped
    if must_be_divisible:
        for i in range(3):
            # Find the largest divisor of bbox_size[i] that is <= chunk_size_clipped[i]
            chunk_size_final[i] = largest_divisor_leq(bbox_size[i], chunk_size_clipped[i])
            if chunk_size_clipped[i] == 0:
                raise ValueError(f"Could not find a valid chunk size for dimension {i}")

    return tuple(chunk_size_final)


def get_num_workers(
    bbox: BBox3D,
    resolutions: Sequence[Sequence[float]] | Sequence[float],
    voxels_per_worker: int = 8 * 8 * 8 * 1024 * 1024 * 4,
) -> int:
    resolution: Sequence[float]
    if isinstance(resolutions[0], float):
        resolution = resolutions  # type: ignore
    else:
        resolution = get_smallest_resolution(resolutions=resolutions)  # type: ignore

    voxel_num = bbox.get_size(resolution=resolution)
    result = math.ceil(voxel_num / voxels_per_worker)
    return result


def get_smallest_resolution(resolutions: Sequence[Sequence[float]]) -> Sequence[float]:
    return sorted([Vec3D(*e) for e in resolutions], key=tuple)[0]


def get_efficient_processing_chunk_size(
    bbox: BBox3D,
    resolution: Sequence[float],
    chunk_size_base: Sequence[int],
    surgery_mode: bool,
    backend_chunk_size: Sequence[int],
    max_task_num: int,
) -> Sequence[int]:
    bbox_slices = bbox.to_slices(resolution=resolution, allow_slice_rounding=True)
    bbox_size = [s.stop - s.start for s in bbox_slices]
    logger.info(f"{resolution} bbox size: {bbox_size}")
    processing_chunk_size = get_chunk_size(
        chunk_size_base=chunk_size_base, bbox_size=bbox_size, must_be_divisible=surgery_mode
    )
    if not surgery_mode:
        processing_chunk_size = [
            math.ceil(processing_chunk_size[i] / backend_chunk_size[i]) * backend_chunk_size[i]
            for i in range(3)
        ]
    logger.info(f"{resolution} processing chunk size: {processing_chunk_size}")
    processing_chunk_size_vx = (
        processing_chunk_size[0] * processing_chunk_size[1] * processing_chunk_size[2]
    )
    bbox_size_vx = bbox_size[0] * bbox_size[1] * bbox_size[2]
    num_tasks_approx = bbox_size_vx / processing_chunk_size_vx
    if num_tasks_approx > max_task_num:
        error_msg = (
            f"Attempting to run a job with more than {max_task_num} tasks "
            "is not allowed with smart self-adjusting portal job due to potential "
            f"high compute costs. Resulting job would have {num_tasks_approx} tasks. "
        )
        if surgery_mode:
            error_msg += (
                "Note that your are using 'surgery', which is inherently inefficient. "
                f"For the specified bounding box of size {bbox_size}, the processing "
                f"size at resolution {resolution} had to be reduced "
                f"to {processing_chunk_size}, as the chunk size has to divide the bbox "
                "evenly. Consider adjusting the bounding box size to be a nicer number. "
            )
        raise RuntimeError(error_msg)
    return processing_chunk_size


def build_simple_multires_flow(
    dst: VolumetricLayer,
    resolutions: Sequence[Sequence[float]],
    bbox: BBox3D,
    write_mode: Literal["extend", "overwrite", "surgery", "efficient_inexact_surgery"],
    precomputed_chunk_size: Sequence[int],
    max_task_num: int,
    op_kwargs: dict,
    fn: Callable | None = None,
    op: VolumetricOpProtocol | None = None,
):
    chunk_size_base = IntVec3D(4 * 1024, 4 * 1024, 32)
    subflows = []
    for res in resolutions:
        processing_chunk_size = get_efficient_processing_chunk_size(
            bbox=bbox,
            resolution=res,
            chunk_size_base=chunk_size_base,
            max_task_num=max_task_num,
            surgery_mode=write_mode == "surgery",
            backend_chunk_size=precomputed_chunk_size,
        )
        subflows.append(
            build_subchunkable_apply_flow(
                bbox=bbox,
                dst_resolution=res,
                processing_chunk_sizes=[processing_chunk_size],
                skip_intermediaries=write_mode != "surgery",
                level_intermediaries_dirs=(
                    ["gs://tmp_2w/subhcunkable_tmp/portal/smart_copy"]
                    if write_mode == "surgery"
                    else None
                ),
                expand_bbox_backend=write_mode == "efficient_inexact_surgery",
                expand_bbox_processing=write_mode != "surgery",
                expand_bbox_resolution=True,
                fn=fn,
                op=op,
                dst=dst,
                op_kwargs=op_kwargs,
            )
        )
    return concurrent_flow(subflows)
