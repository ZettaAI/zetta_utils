from __future__ import annotations

import time
from typing import Callable, Sequence

import attrs
import requests
import yaml
from cloudfiles import CloudFile

from zetta_utils import builder, log, mazepa

logger = log.get_logger("zetta_utils")
from zetta_utils.geometry import BBox3D
from zetta_utils.mazepa import dryrun


def group_consecutive(zs: Sequence[int]) -> list[tuple[int, int]]:
    """Given ints, return (start, end_exclusive) for each consecutive run.
    Sorts and dedups input. [0,1,2,5,6,100] -> [(0,3), (5,7), (100,101)]
    """
    if not zs:
        return []
    zs = sorted(set(zs))
    runs = []
    run_start = zs[0]
    prev = zs[0]
    for z in zs[1:]:
        if z == prev + 1:
            prev = z
            continue
        runs.append((run_start, prev + 1))
        run_start = z
        prev = z
    runs.append((run_start, prev + 1))
    return runs


def _progress_path_for(layer_path: str, resolution_nm: int) -> str:
    return f"{layer_path.rstrip('/')}/_progress/{resolution_nm}nm.progress.yaml"


def _read_progress(progress_path: str) -> set[int]:
    cf = CloudFile(progress_path)
    if not cf.exists():
        return set()
    raw = cf.get()
    if raw is None:
        return set()
    data = yaml.safe_load(raw.decode("utf-8")) or {}
    return set(int(z) for z in data.get("completed", []))


def _write_progress(progress_path: str, completed: set[int]) -> None:
    payload = yaml.safe_dump({"completed": sorted(completed)})
    CloudFile(progress_path).put(payload.encode("utf-8"), content_type="application/yaml")


def _mark_done_all(progress_paths: Sequence[str], all_z_values: Sequence[int]) -> None:
    if dryrun.in_dryrun():
        return
    for progress_path in progress_paths:
        for attempt in range(5):
            try:
                existing = _read_progress(progress_path)
                merged = existing | set(all_z_values)
                if merged != existing:
                    _write_progress(progress_path, merged)
                break
            except (OSError, yaml.YAMLError, requests.exceptions.RequestException):
                if attempt == 4:
                    raise
                time.sleep(1.0 * (2 ** attempt))


@mazepa.flow_schema_cls
@attrs.mutable
class _TrackedSectionFlowSchema:
    compute_flow: mazepa.Flow
    progress_paths: list[str]
    all_z_values: list[int]

    def flow(self):
        yield [self.compute_flow]
        yield mazepa.Dependency()
        _mark_done_all(self.progress_paths, self.all_z_values)


def _narrow_z_bbox(bbox: BBox3D, z_start: int, z_end: int, resolution: Sequence[float]) -> BBox3D:
    """New BBox3D with same xy, z range narrowed to [z_start, z_end) voxels at given resolution."""
    (x0, x1), (y0, y1), _ = bbox.bounds
    z_res_nm = float(resolution[2])
    return BBox3D.from_coords(
        start_coord=(x0, y0, z_start * z_res_nm),
        end_coord=(x1, y1, z_end * z_res_nm),
        resolution=(1, 1, 1),
    )


@builder.register("build_tracked_section_flow", allow_parallel=False)
def build_tracked_section_flow(
    flow_factory: Callable[..., mazepa.Flow],
    layer_resolution_pairs: list[tuple[str, int]],
    bbox: BBox3D,
    resolution: Sequence[float],
    skip_existing: bool,
) -> mazepa.Flow:
    if not callable(flow_factory):
        raise TypeError(f"flow_factory must be callable, got {type(flow_factory)}")

    unique_pairs: list[tuple[str, int]] = list(dict.fromkeys(layer_resolution_pairs))
    progress_paths: list[str] = [
        _progress_path_for(layer_path, resolution_nm) for layer_path, resolution_nm in unique_pairs
    ]

    z_slice = bbox.get_slice(dim=2, resolution=resolution, allow_slice_rounding=False)
    z_start, z_end = int(z_slice.start), int(z_slice.stop)
    all_z = range(z_start, z_end)

    if skip_existing and progress_paths:
        per_layer: list[tuple[str, int, set[int]]] = []
        for path, (layer_path, res_nm) in zip(progress_paths, unique_pairs):
            done = _read_progress(path)
            done_in_range = done & set(all_z)
            per_layer.append((layer_path, res_nm, done_in_range))
            logger.info(
                f"build_tracked_section_flow skip_existing: "
                f"{len(done_in_range)}/{len(all_z)} z already done for "
                f"{layer_path}@{res_nm}nm"
            )
        completed: set[int] = set.intersection(*(d for _, _, d in per_layer))
        logger.info(
            f"build_tracked_section_flow skip_existing: "
            f"{len(completed)}/{len(all_z)} z skipped (intersection across "
            f"{len(progress_paths)} layer(s))"
        )
    else:
        completed = set()
    remaining = [z for z in all_z if z not in completed]
    runs = group_consecutive(remaining)

    if not runs:
        return mazepa.concurrent_flow([])

    per_run_flows: list[mazepa.Flow] = []
    all_z_values: list[int] = []
    for run_start, run_end in runs:
        run_bbox = _narrow_z_bbox(bbox, run_start, run_end, resolution)
        per_run_flows.append(flow_factory(bbox=run_bbox))
        all_z_values.extend(range(run_start, run_end))

    compute_all = mazepa.concurrent_flow(per_run_flows)
    return _TrackedSectionFlowSchema(
        compute_flow=compute_all,
        progress_paths=progress_paths,
        all_z_values=all_z_values,
    )()
