from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

from zetta_utils import builder
from zetta_utils.geometry.bbox import BBox3D
from zetta_utils.geometry.vec import Vec3D
from zetta_utils.internal.portal_jobs.common import (
    build_simple_multires_flow,
    get_num_workers,
)
from zetta_utils.layer.volumetric.cloudvol.build import build_cv_layer
from zetta_utils.layer.volumetric.index import VolumetricIndex
from zetta_utils.layer.volumetric.layer import VolumetricLayer
from zetta_utils.log import get_logger
from zetta_utils.mazepa.tasks import taskable_operation_cls
from zetta_utils.mazepa_addons.configurations.execute_on_gcp_with_sqs import (
    execute_on_gcp_with_sqs,
)

logger = get_logger("zetta_utils")


@taskable_operation_cls
class MakeSegmentMaskOp:
    def get_operation_name(self) -> str:
        return "MakeSegmentMaskOp"

    def with_added_crop_pad(
        self, crop_pad: Vec3D[int]  # pylint: disable=unused-argument
    ) -> MakeSegmentMaskOp:
        return self

    def get_input_resolution(self, dst_resolution: Vec3D[float]) -> Vec3D[float]:
        return dst_resolution

    def __call__(
        self,
        idx: VolumetricIndex,
        dst: VolumetricLayer,
        src: VolumetricLayer,
        selected_segments: Sequence[int],
        keep_existing_values: bool,
        value: int,
    ):
        src_data = src[idx]
        mask = np.isin(src_data, selected_segments)

        result_data = dst[idx]
        if not keep_existing_values:
            result_data = np.zeros_like(result_data)
        result_data[mask] = value
        dst[idx] = result_data


@builder.register("smart_make_segment_mask")
def smart_make_segment_mask(
    src_path: str,
    dst_path: str,
    resolutions: Sequence[Sequence[float]],
    bbox: BBox3D,
    precomputed_chunk_size: Sequence[int],
    write_mode: Literal["extend", "overwrite", "surgery", "efficient_inexact_surgery"],
    selected_segments: Sequence[int],
    keep_existing_values: bool,
    value: float,
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
        info_data_type="uint8",
        info_type="segmentation",
        info_num_channels=1,
        info_encoding="raw",
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
        op=MakeSegmentMaskOp(),
        max_task_num=max_task_num,
        precomputed_chunk_size=precomputed_chunk_size,
        op_kwargs={
            "selected_segments": selected_segments,
            "src": src,
            "keep_existing_values": keep_existing_values,
            "value": value,
        },
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
