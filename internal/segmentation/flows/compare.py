import itertools
import multiprocessing
import os
from typing import Sequence

from tqdm import tqdm
from numpy import typing as npt
import numpy as np
from collections import defaultdict

from zetta_utils import builder
from zetta_utils.db_annotations.annotation import add_bbox_annotation
from zetta_utils.db_annotations.collection import add_collection, read_collection
from zetta_utils.db_annotations.deletion import cascade_delete_collections
from zetta_utils.db_annotations.layer import add_layer
from zetta_utils.db_annotations.layer_group import add_layer_group
from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric.index import VolumetricIndex
from zetta_utils.layer.volumetric.layer import VolumetricLayer
from zetta_utils.mazepa import Dependency, flow_schema_cls
from zetta_utils.mazepa_layer_processing.common.subchunkable_apply_flow import (
    _expand_bbox_processing,
    _expand_bbox_resolution,
)
from zetta_utils.mazepa_layer_processing.common.volumetric_apply_flow import (
    VolumetricApplyFlowSchema,
)
from zetta_utils.mazepa_layer_processing.common.volumetric_callable_operation import (
    VolumetricCallableOperation,
)
from zetta_utils.log import get_logger
from zetta_utils.task_management.project import get_collection

logger = get_logger("zetta_utils")


@flow_schema_cls
class SegCompareFlowSchema:
    def flow(
        self,
        reference_layer: VolumetricLayer,
        candidate_layer: VolumetricLayer,
        bbox: BBox3D,
        resolution: Sequence[float],
        processing_chunk_size: Sequence[int],
        collection_prefix: str | None = None,
        overwrite_collections: bool = False,
        filter_top_n: int = 256,
        expand_bbox_resolution: bool = True,
        expand_bbox_processing: bool = True,
        worker_type: str | None = None,
        min_vx_diff: int = 100,
    ):
        assert "ZETTA_USER" in os.environ, "ZETTA_USER must be set"
        user = os.environ["ZETTA_USER"]
        if collection_prefix is not None:
            for suffix in ["mergers", "splits"]:
                try:
                    read_collection(collection_prefix + "_" + suffix)
                except KeyError:
                    ...
                else:
                    if not overwrite_collections:
                        raise ValueError(f"Collection {collection_prefix}_mergers already exists")
                    else:
                        cascade_delete_collections([collection_prefix + "_" + suffix])

        resolution_vec = Vec3D[float](*resolution)
        processing_chunk_size_vec = Vec3D[int](*processing_chunk_size)
        if expand_bbox_resolution:
            bbox = _expand_bbox_resolution(bbox, resolution_vec)

        elif expand_bbox_processing:
            bbox = _expand_bbox_processing(
                bbox, resolution_vec, [processing_chunk_size_vec], Vec3D[int](0, 0, 0)
            )
        flow = VolumetricApplyFlowSchema(
            op=VolumetricCallableOperation(seg_compare, fn_semaphores=["read", "write"]),
            processing_chunk_size=processing_chunk_size_vec,
            dst_resolution=resolution_vec,
            op_worker_type=worker_type,
        )(
            idx=VolumetricIndex(bbox=bbox, resolution=resolution_vec),
            dst=None,
            op_kwargs={
                "reference": reference_layer,
                "candidate": candidate_layer,
                "min_vx_diff": min_vx_diff,
            },
            op_args=[],
        )
        tasks = flow.get_next_batch()
        while True:
            new_tasks = flow.get_next_batch()
            if new_tasks is None:
                break
            tasks.extend(new_tasks)
        yield tasks
        yield Dependency()

        combined_splits = defaultdict(lambda: defaultdict(int))
        combined_mergers = defaultdict(lambda: defaultdict(int))

        for task in tasks:
            # dryrun
            if task.outcome is None:
                return

            for ref_id, candidates in task.outcome.return_value["splits"].items():
                for cand_id, count in candidates.items():
                    combined_splits[int(ref_id)][int(cand_id)] += int(count)

            for cand_id, references in task.outcome.return_value["merges"].items():
                for ref_id, count in references.items():
                    combined_mergers[int(cand_id)][int(ref_id)] += int(count)
        logger.info(f"Found {len(combined_splits)} splits and {len(combined_mergers)} mergers")
        split_sizes = {
            ref_id: sum(counts.values()) - max(counts.values()) if counts else 0
            for ref_id, counts in combined_splits.items()
        }
        merger_sizes = {
            cand_id: sum(counts.values()) - max(counts.values()) if counts else 0
            for cand_id, counts in combined_mergers.items()
        }
        def get_top_items(size_dict, max_items=10):
            return dict(sorted(size_dict.items(), key=lambda x: x[1], reverse=True)[:max_items])

        top_splits = {
            ref_id: list(combined_splits[ref_id].keys())
            for ref_id in get_top_items(split_sizes, filter_top_n)
        }
        top_mergers = {
            cand_id: list(combined_mergers[cand_id].keys())
            for cand_id in get_top_items(merger_sizes, filter_top_n)
        }
        logger.info(f"Top splits: {top_splits}")
        logger.info(f"Top mergers: {top_mergers}")
        if collection_prefix is not None:

            for mode in ["splits", "mergers"]:
                collection_id = add_collection(
                    collection_prefix + "_" + mode,
                    user,
                    comment=f"{mode} from {reference_layer.name} to {candidate_layer.name}",
                )
                reference_layer_id = add_layer(name="reference", source=reference_layer.name)
                candidate_layer_id = add_layer(name="candidate", source=candidate_layer.name)
                layer_group_id = add_layer_group(
                    name=f"Layers",
                    collection_id=collection_id,
                    user=user,
                    layers=[reference_layer_id, candidate_layer_id],
                )
                breakpoint()
                if mode == "splits":
                    for ref_id, cand_ids in tqdm(top_splits.items(), desc=f"Adding {mode} annotations"):
                        selected_segments = {
                            "reference": [ref_id],
                            "candidate": cand_ids,
                        }
                        add_bbox_annotation(
                            bbox=bbox,
                            collection_id=collection_id,
                            layer_group_id=layer_group_id,
                            selected_segments=selected_segments,
                            comment=f"{ref_id} split"
                        )
                else:
                    for cand_id, ref_ids in tqdm(top_mergers.items(), desc=f"Adding {mode} annotations"):
                        selected_segments = {
                            "reference": ref_ids,
                            "candidate": [cand_id],
                        }
                        add_bbox_annotation(
                            bbox=bbox,
                            collection_id=collection_id,
                            layer_group_id=layer_group_id,
                            selected_segments=selected_segments,
                            comment=f"{cand_id} merge"
                        )

def seg_compare(reference: npt.NDArray, candidate: npt.NDArray, min_vx_diff: int):
    def _find_segment_overlaps(
        source_array: npt.NDArray,
        target_array: npt.NDArray,
        min_vx_diff: int,
    ) -> dict:
        result_dict = {}
        source_segments = np.unique(source_array)

        for source_seg in source_segments:
            if source_seg == 0:
                continue

            mask = source_array == source_seg
            unique_values, counts = np.unique(target_array[mask], return_counts=True)
            result_dict[source_seg] = {
                val: count
                for val, count in zip(unique_values, counts)
                if count > min_vx_diff and val != 0
            }

            if len(result_dict[source_seg]) <= 1:
                del result_dict[source_seg]

        return {k: dict(v) for k, v in result_dict.items()}

    result = {
        "splits": _find_segment_overlaps(
            source_array=reference,
            target_array=candidate,
            min_vx_diff=min_vx_diff,
        ),
        "merges": _find_segment_overlaps(
            source_array=candidate,
            target_array=reference,
            min_vx_diff=min_vx_diff,
        ),
    }
    return result


@builder.register("build_seg_compare_flow")
def build_seg_compare_flow(
    reference_layer: VolumetricLayer,
    candidate_layer: VolumetricLayer,
    bbox: BBox3D,
    resolution: Sequence[float],
    processing_chunk_size: Sequence[int],
    collection_prefix: str | None = None,
    overwrite_collections: bool = False,
    filter_top_n: int = 256,
):
    return SegCompareFlowSchema()(
        reference_layer=reference_layer,
        candidate_layer=candidate_layer,
        bbox=bbox,
        resolution=resolution,
        processing_chunk_size=processing_chunk_size,
        collection_prefix=collection_prefix,
        overwrite_collections=overwrite_collections,
        filter_top_n=filter_top_n,
    )
