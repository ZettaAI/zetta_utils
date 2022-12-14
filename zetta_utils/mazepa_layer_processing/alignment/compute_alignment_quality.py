import datetime
from statistics import NormalDist
from typing import Any, Dict, List, TypeVar

from typeguard import typechecked
from typing_extensions import ParamSpec

from zetta_utils import builder, log, mazepa
from zetta_utils.layer.volumetric import (
    VolumetricIndex,
    VolumetricIndexChunker,
    VolumetricLayer,
)
from zetta_utils.mazepa import Dependency
from zetta_utils.typing import IntVec3D, Vec3D

logger = log.get_logger("zetta_utils")

IndexT = TypeVar("IndexT")
P = ParamSpec("P")
R_co = TypeVar("R_co", covariant=True)


@mazepa.taskable_operation
def compute_misalignment_stats(
    layer: VolumetricLayer,
    idx: VolumetricIndex,
    misalignment_thresholds: List[float],
) -> Dict[str, Any]:
    data = layer[idx]
    ret = {}  # type: Dict[str, Any]
    ret["sum"] = data.sum()
    ret["sqsum"] = (data ** 2).sum()
    ret["size"] = data.nelement()
    ret["misaligned_pixels"] = {}
    for threshold in misalignment_thresholds:
        ret["misaligned_pixels"][threshold] = (data > threshold).sum()
    return ret


def lrpad(
    string: str = "", level: int = 1, length: int = 80, bounds: str = "|", filler: str = " "
) -> str:
    newstr = ""
    newstr += bounds
    while len(newstr) < level * 4:
        newstr += filler
    newstr += string
    if len(newstr) >= length:
        return newstr
    while len(newstr) < length - 1:
        newstr += filler
    return newstr + bounds


# f-string-without-interpolation should not be necessary, but pylint seems to have a bug
@builder.register(
    "compute_alignment_quality", cast_to_vec3d=["resolution"], cast_to_intvec3d=["chunk_size"]
)
@mazepa.flow_schema
@typechecked
def compute_alignment_quality(
    src: VolumetricLayer,
    idx: VolumetricIndex,
    chunk_size: IntVec3D,
    resolution: Vec3D,
    misalignment_thresholds: List[float],
    num_worst_chunks: int,
):  # pylint: disable = too-many-locals, too-many-statements, too-many-branches, f-string-without-interpolation
    chunker = VolumetricIndexChunker(chunk_size)

    logger.info(f"Breaking {idx} into chunks with {chunker}.")
    idx_chunks = list(chunker(idx))
    tasks = [
        compute_misalignment_stats.make_task(
            idx=idx_chunk,
            layer=src,
            misalignment_thresholds=misalignment_thresholds,
        )
        for idx_chunk in idx_chunks
    ]
    if num_worst_chunks > len(tasks):
        logger.error(
            f"{num_worst_chunks} worst chunks requested," + f"but only {len(tasks)} chunks exist."
        )
    logger.info(f"Submitting {len(tasks)} processing tasks from factory.")

    yield tasks
    yield Dependency()

    sums = []
    sqsums = []
    sizes = []
    rmses = []
    worsts = []
    misaligned_pixelss = {}  # type: Dict[float, Any]
    misaligned = {}  # type: Dict[float, Any]

    for threshold in misalignment_thresholds:
        misaligned_pixelss[threshold] = []

    # parse outputs
    for task in tasks:
        assert task.outcome is not None
        ret = task.outcome.return_value
        assert ret is not None
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

    print(lrpad("  Alignment Quality Report  ", bounds="+", filler="="))
    print(lrpad())
    print(lrpad(f"Generated {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}", 1))
    print(lrpad())
    print(lrpad(" Dataset Information / Settings ", bounds="|", filler="="))
    print(lrpad())
    print(lrpad(f"Layer: {src.get_name()}", 1))
    print(lrpad(f"Bounds given at {resolution} nm: {idx.pformat(resolution)}", 1))

    vol = idx.get_size()

    if vol < 1e18:
        print(lrpad(f"Volume of FOV: {vol*1e-9:10.3f} um^3", 1))
    else:
        print(lrpad(f"Volume of FOV: {vol*1e-18:10.3f} mm^3", 1))
    print(lrpad(f"Misalignment detection resolution: {idx.resolution} nm", 1))
    print(lrpad(f"Misalignment thresholds:", 1))
    for threshold in misalignment_thresholds:
        print(lrpad(f"{threshold:3.2f} px ({threshold * idx.resolution[0]:6.2f} nm)", 2))
    print(lrpad(f"Number of worst chunks to show: {num_worst_chunks}", 1))
    print(lrpad())

    print(lrpad(" Basic Misalignment Statistics ", bounds="|", filler="="))
    print(lrpad())
    print(lrpad(f"RMS residuals:        {rms:7.4f} px ({rms * idx.resolution[0]:7.4f} nm)", 1))
    print(lrpad(f"Mean residuals:       {mean:7.4f} px ({mean * idx.resolution[0]:7.4f} nm)", 1))
    print(lrpad(f"Probability of misaligned pixel at ", 1))
    for threshold in misalignment_thresholds:
        misaligned_prob = misaligned[threshold]["probability"]
        if misaligned_prob == 0.0:
            misaligned_sigma_str = "infty"
        else:
            misaligned_sigma_str = f"{abs(NormalDist().inv_cdf(misaligned_prob)):2.3f}"
        print(
            lrpad(
                f"{threshold:3.2f} px: {(misaligned_prob * 1e6):10.3f}"
                + " parts per million,      "
                + f"{misaligned_sigma_str} sigmas",
                2,
            )
        )
    print(lrpad())

    print(lrpad(" Proofreading Helper ", bounds="|", filler="="))
    print(lrpad())
    print(lrpad(f"Worst chunk(s) overall (RMS):", 1))
    for i in range(num_worst_chunks):
        ind = worsts[i]
        print(lrpad(f"{idx_chunks[ind].pformat(resolution)}       {rmses[ind]:7.4f} px", 2))
    for threshold in misalignment_thresholds:
        print(lrpad(f"Worst chunk(s) at {threshold:3.2f} pixels:", 1))
        for i in range(num_worst_chunks):
            ind = misaligned[threshold]["worsts"][i]
            prob = misaligned_pixelss[threshold][ind] / sizes[ind]
            print(lrpad(f"{idx_chunks[ind].pformat(resolution)}   {(prob * 1e6):10.3f} ppm", 2))
    print(lrpad(""))
    print(lrpad("", bounds="+", filler="="))
