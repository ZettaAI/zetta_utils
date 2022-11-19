import copy
import os
from typing import Any, Callable, Generic, List, Literal, TypeVar

import attrs
import torch
from typeguard import typechecked
from typing_extensions import ParamSpec

from zetta_utils import builder, log, mazepa
from zetta_utils.layer import IndexChunker, Layer, LayerIndex
from zetta_utils.layer.volumetric import VolIdxTranslator, VolumetricIndex
from zetta_utils.mazepa import Dependency, task_factory

from .. import ChunkedApplyFlowType, SimpleCallableTaskFactory

logger = log.get_logger("zetta_utils")

IndexT = TypeVar("IndexT", bound=LayerIndex)
P = ParamSpec("P")
R_co = TypeVar("R_co", covariant=True)


@task_factory
def compute_misalignment_stats(**kwargs):
    data = kwargs["data"][kwargs["idx"]]
    misalignment_thresholds = kwargs["misalignment_thresholds"]
    #    print(f"The mean is {data.mean()}")
    ret = {}
    ret["mean"] = data.mean()
    ret["size"] = data.nelement()
    ret["misaligned"] = {}
    for threshold in misalignment_thresholds:
        ret["misaligned"][threshold] = (data > threshold).sum()
    return ret

    # TODO: Separate


def pretty_print(idx, resolution):
    slices = idx.bcube.to_slices(resolution)
    x_start = slices[0].start
    x_end = slices[0].stop
    y_start = slices[1].start
    y_end = slices[1].stop
    z_start = slices[2].start
    z_end = slices[2].stop
    return f"{x_start}-{x_end}, {y_start}-{y_end}, {z_start}-{z_end}"


@builder.register("compute_alignment_quality")
@mazepa.flow_type
@typechecked
def compute_alignment_quality(
    src: Layer[Any, VolumetricIndex, torch.Tensor],
    idx: VolumetricIndex,
    chunker: IndexChunker[VolumetricIndex],
    # TODO: fix this typing
    resolution: List[Any],
    misalignment_thresholds: List[Any],
    #    object_size: Any
):
    #    task_kwargs = {k: v for k, v in kwargs.items() if k not in ["idx"]}
    logger.info(f"Breaking {idx} into chunks with {chunker}.")
    idx_chunks = chunker(idx)
    tasks = [
        compute_misalignment_stats.make_task(
            idx=idx_chunk,  # type: ignore
            data=src,
            misalignment_thresholds=misalignment_thresholds
            #            object_size=object_size
        )
        for idx_chunk in idx_chunks
    ]
    logger.info(f"Submitting {len(tasks)} processing tasks from factory.")

    for task in tasks:
        yield task
    yield Dependency()

    means = []
    sizes = []
    misaligneds = {}

    for threshold in misalignment_thresholds:
        misaligneds[threshold] = []

    # parse outputs
    for task in tasks:
        ret = task.outcome.return_value
        means.append(ret["mean"])
        sizes.append(ret["size"])
        for threshold in misalignment_thresholds:
            misaligneds[threshold].append(ret["misaligned"][threshold])

    mean = sum(means) / len(means)
    worst_ind = means.index(max(means))
    size_total = sum(sizes)
    misaligned_totals = {}
    misaligned_maxinds = {}
    for threshold in misalignment_thresholds:
        misaligned_totals[threshold] = sum(misaligneds[threshold])
        misaligned_maxinds[threshold] = misaligneds[threshold].index(max(misaligneds[threshold]))

    print(f"===================================================================================")
    print(f"===   Alignment Quality Index Summary     =========================================")
    print(f"===================================================================================")
    print(f"===== Dataset Information / Settings ==============================================")

    # TODO: Hardcoded name path
    print(f"=   Layer: {src.backend.path}")

    print(f"=   Bounds at {resolution} nm: {pretty_print(idx, resolution)}")

    # TODO: Separate
    slices = idx.bcube.to_slices([1, 1, 1])
    x_start = slices[0].start
    x_end = slices[0].stop
    y_start = slices[1].start
    y_end = slices[1].stop
    z_start = slices[2].start
    z_end = slices[2].stop
    vol = (x_end - x_start) * (y_end - y_start) * (z_end - z_start) * 1e-9

    print(f"=   Volume of FOV: {vol:10.3f} um^3")
    print(f"=   Misalignment detection resolution: {idx.resolution} nm")
    print(f"=   Misalignment thresholds:")
    for threshold in misalignment_thresholds:
        print(f"=             {threshold:3.2f} px ({threshold * idx.resolution[0]:6.2f} nm)")
    print(f"=")

    print(f"===== Basic Misalignment Statistics ===============================================")

    print(f"=   Mean residuals:       {mean:7.4f} px ({mean * idx.resolution[0]:7.4f} nm)")
    print(f"=   Misaligned at ")
    for threshold in misalignment_thresholds:
        print(
            f"=                 {threshold:3.2f} px: {(misaligned_totals[threshold] / size_total * 1e6):10.3f} parts per million"
        )
    print(f"=")

    print(f"===== Proofreading Helper =========================================================")
    print(f"=   Worst chunk overall: {pretty_print(idx_chunks[worst_ind], resolution)}")
    print(f"=   Worst chunk at")
    for threshold in misalignment_thresholds:
        print(
            f"=             {threshold:3.2f} px: {pretty_print(idx_chunks[misaligned_maxinds[threshold]], resolution)}"
        )
    print(f"=")
    print(f"===================================================================================")
