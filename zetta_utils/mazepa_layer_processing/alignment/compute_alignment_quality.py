import copy
import datetime
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


class TablePrinter(object):
    "Print a list of dicts as a table"

    def __init__(self, fmt, sep=" ", ul=None):
        """
        @param fmt: list of tuple(heading, key, width)
                        heading: str, column label
                        key: dictionary key to value to print
                        width: int, column width in chars
        @param sep: string, separation between columns
        @param ul: string, character to underline column label, or None for no underlining
        """
        super(TablePrinter, self).__init__()
        self.fmt = str(sep).join(
            "{lb}{0}:{1}{rb}".format(key, width, lb="{", rb="}") for heading, key, width in fmt
        )
        self.head = {key: heading for heading, key, width in fmt}
        self.ul = {key: str(ul) * width for heading, key, width in fmt} if ul else None
        self.width = {key: width for heading, key, width in fmt}

    def row(self, data):
        return self.fmt.format(**{k: str(data.get(k, ""))[:w] for k, w in self.width.iteritems()})

    def __call__(self, dataList):
        _r = self.row
        res = [_r(data) for data in dataList]
        res.insert(0, _r(self.head))
        if self.ul:
            res.insert(1, _r(self.ul))
        return "\n".join(res)


@task_factory
def compute_misalignment_stats(**kwargs):
    data = kwargs["data"][kwargs["idx"]]
    misalignment_thresholds = kwargs["misalignment_thresholds"]
    #    print(f"The mean is {data.mean()}")
    ret = {}
    ret["sum"] = data.sum()
    ret["sqsum"] = (data ** 2).sum()
    ret["size"] = data.nelement()
    ret["misaligned_pixels"] = {}
    for threshold in misalignment_thresholds:
        ret["misaligned_pixels"][threshold] = (data > threshold).sum()
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


def lrpad(string="", level=1, length=80, char="|"):
    newstr = ""
    newstr += char
    while len(newstr) < level * 4:
        newstr += " "
    newstr += string
    if len(newstr) >= length:
        return newstr
    else:
        while len(newstr) < length - 1:
            newstr += " "
        return newstr + char


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
    num_worst_chunks: int,
):
    #    task_kwargs = {k: v for k, v in kwargs.items() if k not in ["idx"]}
    logger.info(f"Breaking {idx} into chunks with {chunker}.")
    idx_chunks = chunker(idx)
    tasks = [
        compute_misalignment_stats.make_task(
            idx=idx_chunk,  # type: ignore
            data=src,
            misalignment_thresholds=misalignment_thresholds,
        )
        for idx_chunk in idx_chunks
    ]
    logger.info(f"Submitting {len(tasks)} processing tasks from factory.")

    for task in tasks:
        yield task
    yield Dependency()

    sums = []
    sqsums = []
    sizes = []
    rmses = []
    worsts = []
    misaligned_pixelss = {}
    misaligned = {}

    for threshold in misalignment_thresholds:
        misaligned_pixelss[threshold] = []

    # parse outputs
    for task in tasks:
        ret = task.outcome.return_value
        sums.append(ret["sum"])
        sqsums.append(ret["sqsum"])
        sizes.append(ret["size"])
        rmses.append((ret["sqsum"] / ret["size"]) ** 0.5)
        for threshold in misalignment_thresholds:
            misaligned_pixelss[threshold].append(ret["misaligned_pixels"][threshold])

    size_total = sum(sizes)
    mean = sum(sums) / size_total
    rms = (sum(sqsums) / size_total) ** 0.5
    inds = list(range(len(rmses)))
    sorted_inds = [ind for _, ind in sorted(zip(rmses, inds), reverse=True)]
    worsts = sorted_inds[0:num_worst_chunks]

    for threshold in misalignment_thresholds:
        misaligned[threshold] = {}
        misaligned[threshold]["threshold"] = threshold
        misaligned[threshold]["probability"] = sum(misaligned_pixelss[threshold]) / size_total

        # sort by the number of misaligned pixels at the resolution
        inds = list(range(len(misaligned_pixelss[threshold])))
        sorted_inds = [
            ind for _, ind in sorted(zip(misaligned_pixelss[threshold], inds), reverse=True)
        ]
        misaligned[threshold]["worsts"] = sorted_inds[0:num_worst_chunks]

    print(f"===   Alignment Quality Report   ===============================================")
    print(lrpad())
    print(lrpad(f"Generated {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}", 1))
    print(lrpad())
    print(f"|==== Dataset Information / Settings ==========================================|")
    print(lrpad())
    print(lrpad(f"Layer: {src.backend.path}", 1))
    # TODO: Hardcoded name path
    print(lrpad(f"Bounds given at {resolution} nm: {pretty_print(idx, resolution)}", 1))

    # TODO: Separate
    slices = idx.bcube.to_slices([1, 1, 1])
    x_start = slices[0].start
    x_end = slices[0].stop
    y_start = slices[1].start
    y_end = slices[1].stop
    z_start = slices[2].start
    z_end = slices[2].stop
    vol = (x_end - x_start) * (y_end - y_start) * (z_end - z_start)

    if vol < 1e-18:
        print(lrpad(f"Volume of FOV: {vol*1e-9:10.3f} um^3", 1))
    else:
        print(lrpad(f"Volume of FOV: {vol*1e-18:10.3f} mm^3", 1))
    print(lrpad(f"Misalignment detection resolution: {idx.resolution} nm", 1))
    print(lrpad(f"Misalignment thresholds:", 1))
    for threshold in misalignment_thresholds:
        print(lrpad(f"{threshold:3.2f} px ({threshold * idx.resolution[0]:6.2f} nm)", 2))
    print(lrpad(f"Number of worst chunks to show: {num_worst_chunks}", 1))
    print(lrpad())

    print(f"|=== Basic Misalignment Statistics ============================================|")
    print(lrpad())
    print(lrpad(f"RMS residuals:        {rms:7.4f} px ({rms * idx.resolution[0]:7.4f} nm)", 1))
    print(lrpad(f"Mean residuals:       {mean:7.4f} px ({mean * idx.resolution[0]:7.4f} nm)", 1))
    print(lrpad(f"Probability of misaligned pixel at ", 1))
    for threshold in misalignment_thresholds:
        print(
            lrpad(
                f"{threshold:3.2f} px: {(misaligned[threshold]['probability'] * 1e6):10.3f} parts per million",
                2
            )
        )
    print(lrpad())

    print(f"|=== Proofreading Helper ======================================================|")
    print(lrpad())
    print(lrpad(f"Worst chunk(s) overall (RMS):", 1))
    for i in range(num_worst_chunks):
        ind = worsts[i]
        print(lrpad(f"{pretty_print(idx_chunks[ind], resolution)}, {rmses[ind]:7.4f} px", 2))
    for threshold in misalignment_thresholds:
        print(lrpad(f"Worst chunk(s) at {threshold:3.2f} pixels:", 1))
        for i in range(num_worst_chunks):
            ind = misaligned[threshold]["worsts"][i]
            prob = misaligned_pixelss[threshold][ind] / sizes[ind]
            print(
                lrpad(f"{pretty_print(idx_chunks[ind], resolution)}, {(prob * 1e6):10.3f} ppm", 2)
            )
    print(lrpad(""))
    print(f"================================================================================")
