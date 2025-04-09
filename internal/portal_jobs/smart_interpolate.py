from typing import Literal, Sequence

from zetta_utils import builder
from zetta_utils.geometry.bbox import BBox3D
from zetta_utils.geometry.vec import IntVec3D
from zetta_utils.internal.portal_jobs.common import (
    get_efficient_processing_chunk_size,
    get_num_workers,
    get_smallest_resolution,
)
from zetta_utils.layer.volumetric.cloudvol.build import build_cv_layer
from zetta_utils.log import get_logger
from zetta_utils.mazepa_addons.configurations.execute_on_gcp_with_sqs import (
    execute_on_gcp_with_sqs,
)
from zetta_utils.mazepa_layer_processing.common.interpolate_flow import (
    build_interpolate_flow,
)
from zetta_utils.tensor_ops.common import InterpolationMode

logger = get_logger("zetta_utils")


def get_interpolation_steps(
    src_resolution: Sequence[float], tgt_resolution: Sequence[float]
) -> list[list[float]]:
    """
    Generate intermediate resolution triplets between a source and target.

    For each dimension:
      - If the source value is less than the target, double it (up to the target).
      - If the source value is greater than the target, halve it (down to the target).

    This is done for all dimensions simultaneously. In any step, the change in any dimension
    is at most a factor of 2. The process continues until each dimension reaches its target.
    """
    tol = 1e-9
    current = list(src_resolution)
    steps = []

    # Continue until every dimension is within tolerance of its target.
    while any(abs(c - t) > tol for c, t in zip(current, tgt_resolution)):
        for i in range(3):
            if current[i] < tgt_resolution[i]:
                current[i] = min(current[i] * 2, tgt_resolution[i])
            elif current[i] > tgt_resolution[i]:
                current[i] = max(current[i] / 2, tgt_resolution[i])
        steps.append(current.copy())

    return steps


@builder.register("smart_interpolate")
def smart_interpolate(
    src_path: str,
    dst_path: str,
    src_resolution: Sequence[float],
    dst_resolution: Sequence[float],
    bbox: BBox3D,
    interpolation_mode: InterpolationMode,
    precomputed_chunk_size: Sequence[int],
    write_mode: Literal["extend", "overwrite", "efficient_inexact_surgery"],
    worker_image: str,
    force_single_worker: bool = False,
    max_num_workers: int = 250,
    max_task_num: int = 300000,
):
    if force_single_worker:
        num_workers = 1
    else:
        num_workers = min(get_num_workers(bbox, src_resolution), max_num_workers)
    logger.info(f"Number of workers: {num_workers}")
    dst_resolutions = get_interpolation_steps(src_resolution, dst_resolution)
    if src_path != dst_path:
        dst_resolutions = [list(src_resolution)] + dst_resolutions
    src = build_cv_layer(path=src_path)
    dst = build_cv_layer(
        path=dst_path,
        info_reference_path=dst_path if write_mode in ["efficient_inexact_surgery"] else src_path,
        info_inherit_all_params=True,
        info_scales=dst_resolutions,
        info_chunk_size=precomputed_chunk_size,
        info_keep_existing_scales=write_mode in ["extend", "efficient_inexact_surgery"],
        info_bbox=bbox if write_mode != "efficient_inexact_surgery" else None,
        info_overwrite=write_mode == "overwrite",
    )
    processing_chunk_size = get_efficient_processing_chunk_size(
        bbox=bbox,
        resolution=get_smallest_resolution(dst_resolutions),
        chunk_size_base=IntVec3D(2 * 1024, 2 * 1024, 32),
        max_task_num=max_task_num,
        surgery_mode=False,
        backend_chunk_size=precomputed_chunk_size,
    )
    target = build_interpolate_flow(
        src=src,
        dst=dst,
        src_resolution=src_resolution,
        dst_resolutions=dst_resolutions,
        bbox=bbox,
        mode=interpolation_mode,
        processing_chunk_sizes=[processing_chunk_size],
        skip_intermediaries=True,
        expand_bbox_backend=write_mode == "efficient_inexact_surgery",
        expand_bbox_processing=True,
        expand_bbox_resolution=True,
    )
    execute_on_gcp_with_sqs(
        target=target,
        worker_image=worker_image,
        worker_groups={
            "all_workers": {
                "replicas": num_workers,
                "resource_limits": {"memory": "32560Mi"},
                "sqs_based_scaling": False,
            }
        },
        local_test=num_workers <= 1,
        worker_cluster_name="zutils-x3",
        worker_cluster_region="us-east1",
        worker_cluster_project="zetta-research",
        write_progress_summary=True,
        require_interrupt_confirm=False,
    )
