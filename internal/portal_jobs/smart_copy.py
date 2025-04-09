from typing import Callable, Literal, Sequence

from zetta_utils import builder
from zetta_utils.geometry.bbox import BBox3D
from zetta_utils.internal.portal_jobs.common import (
    build_simple_multires_flow,
    get_num_workers,
)
from zetta_utils.layer.volumetric.cloudvol.build import build_cv_layer
from zetta_utils.log import get_logger
from zetta_utils.mazepa_addons.configurations.execute_on_gcp_with_sqs import (
    execute_on_gcp_with_sqs,
)

logger = get_logger("zetta_utils")


@builder.register("smart_copy")
def smart_copy(
    src_path: str,
    dst_path: str,
    resolutions: Sequence[Sequence[float]],
    bbox: BBox3D,
    precomputed_chunk_size: Sequence[int],
    write_mode: Literal["extend", "overwrite", "surgery", "efficient_inexact_surgery"],
    fn: Callable,
    worker_image: str,
    force_single_worker: bool,
    max_num_workers: int = 250,
    max_task_num: int = 300000,
):
    if force_single_worker:
        num_workers = 1
    else:
        num_workers = min(get_num_workers(bbox, resolutions), max_num_workers)
    logger.info(f"Number of workers: {num_workers}")
    src = build_cv_layer(path=src_path)
    dst = build_cv_layer(
        path=dst_path,
        info_reference_path=dst_path
        if write_mode in ["surgery", "efficient_inexact_surgery"]
        else src_path,
        info_inherit_all_params=True,
        info_scales=resolutions,
        info_chunk_size=precomputed_chunk_size,
        info_keep_existing_scales=write_mode in ["surgery", "extend", "efficient_inexact_surgery"],
        info_bbox=bbox if write_mode in ["extend", "overwrite"] else None,
        info_overwrite=write_mode == "overwrite",
    )
    target = build_simple_multires_flow(
        dst=dst,
        resolutions=resolutions,
        bbox=bbox,
        write_mode=write_mode,
        fn=fn,
        max_task_num=max_task_num,
        precomputed_chunk_size=precomputed_chunk_size,
        op_kwargs={"src": src},
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
