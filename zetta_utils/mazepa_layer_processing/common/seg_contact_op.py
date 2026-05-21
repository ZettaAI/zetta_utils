from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import attrs
import cc3d
import numpy as np
import pandas as pd
import trimesh
from cloudvolume import CloudVolume
from cloudvolume.exceptions import MeshDecodeError

try:
    from cloudvolume.exceptions import MeshMissingError
except ImportError:
    MeshMissingError = MeshDecodeError
from scipy.spatial.distance import cdist

from zetta_utils import builder, log
from zetta_utils.geometry import Vec3D
from zetta_utils.layer.precomputed import get_info
from zetta_utils.layer.volumetric import VolumetricIndex, VolumetricLayer
from zetta_utils.layer.volumetric.seg_contact import (
    SegContact,
    VolumetricSegContactLayer,
)
from zetta_utils.mazepa import taskable_operation_cls
from zetta_utils.mazepa.semaphores import semaphore

logger = log.get_logger("zetta_utils")


def _read_layers_parallel(
    segmentation: VolumetricLayer,
    reference: VolumetricLayer,
    affinity: VolumetricLayer,
    idx: VolumetricIndex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read segmentation, reference, and affinity layers in parallel."""

    def read_layer(layer: VolumetricLayer, index: VolumetricIndex) -> np.ndarray:
        return np.asarray(layer[index])

    with ThreadPoolExecutor(max_workers=3) as executor:
        seg_future = executor.submit(read_layer, segmentation, idx)
        ref_future = executor.submit(read_layer, reference, idx)
        aff_future = executor.submit(read_layer, affinity, idx)
        return seg_future.result().squeeze(), ref_future.result().squeeze(), aff_future.result()


def _compute_overlaps(
    seg: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute overlaps between segments and reference connected components."""
    cc_ref = cc3d.connected_components(reference, connectivity=6)
    flat_seg = seg.ravel()
    flat_cc_ref = cc_ref.ravel()
    valid_mask = (flat_seg != 0) & (flat_cc_ref != 0)
    df = pd.DataFrame({"seg": flat_seg[valid_mask], "cc_ref": flat_cc_ref[valid_mask]})
    counts_df = df.groupby(["seg", "cc_ref"]).size().reset_index(name="count")
    return (
        counts_df["seg"].values.astype(np.int64),
        counts_df["cc_ref"].values.astype(np.int64),
        counts_df["count"].values.astype(np.int32),
    )


def _find_small_segment_ids(seg: np.ndarray, min_seg_size_vx: int) -> set[int]:
    """Find segment IDs with total voxel count below threshold."""
    unique, counts = np.unique(seg, return_counts=True)
    return {int(s) for s, cnt in zip(unique, counts) if s != 0 and cnt < min_seg_size_vx}


def _find_merger_segment_ids(
    seg_ids: np.ndarray, ref_ids: np.ndarray, counts: np.ndarray, min_overlap_vx: int
) -> set[int]:
    """Find merger segments (overlap 2+ reference CCs with >= min_overlap each)."""
    seg_to_ref: dict[int, set[int]] = defaultdict(set)
    for seg, ref, cnt in zip(seg_ids, ref_ids, counts):
        if cnt >= min_overlap_vx:
            seg_to_ref[int(seg)].add(int(ref))
    return {seg for seg, refs in seg_to_ref.items() if len(refs) >= 2}


def _find_unclaimed_segment_ids(
    seg_ids: np.ndarray, counts: np.ndarray, min_overlap_vx: int, all_seg_ids: set[int]
) -> set[int]:
    """Find segments without sufficient reference overlap.

    This includes both:
    1. Segments with some overlap but below min_overlap_vx threshold
    2. Segments with ZERO overlap (not present in seg_ids at all)
    """
    seg_max_overlap: dict[int, int] = defaultdict(int)
    for seg, cnt in zip(seg_ids, counts):
        seg_max_overlap[int(seg)] = max(seg_max_overlap[int(seg)], int(cnt))

    # Segments with insufficient overlap
    insufficient_overlap = {
        seg for seg, max_ovl in seg_max_overlap.items() if max_ovl < min_overlap_vx
    }

    # Segments with ZERO overlap (not in seg_ids at all)
    segs_with_any_overlap = set(int(s) for s in seg_ids)
    zero_overlap = all_seg_ids - segs_with_any_overlap - {0}

    return insufficient_overlap | zero_overlap


def _find_offtarget_segment_ids(
    seg_ids: np.ndarray,
    ref_ids: np.ndarray,
    counts: np.ndarray,
    seg_total_vx: dict[int, int],
    max_offtarget_vx: int | None,
    max_offtarget_fraction: float | None,
    includes_unclaimed: bool,
) -> set[int]:
    """Find segments with too many off-target voxels.

    Off-target voxels are those overlapping non-best reference CCs.
    When includes_unclaimed is True, also counts voxels with no GT (reference == 0).
    """
    seg_best: dict[int, tuple[int, int]] = {}
    for seg, ref, cnt in zip(seg_ids, ref_ids, counts):
        s, c = int(seg), int(cnt)
        if s not in seg_best or c > seg_best[s][1]:
            seg_best[s] = (int(ref), c)

    seg_total_overlap: dict[int, int] = defaultdict(int)
    for seg, cnt in zip(seg_ids, counts):
        seg_total_overlap[int(seg)] += int(cnt)

    result: set[int] = set()
    for s, total in seg_total_vx.items():
        best_count = seg_best[s][1] if s in seg_best else 0
        covered = seg_total_overlap.get(s, 0)
        offtarget = (total - best_count) if includes_unclaimed else (covered - best_count)
        if max_offtarget_vx is not None and offtarget > max_offtarget_vx:
            result.add(s)
        elif (
            max_offtarget_fraction is not None
            and total > 0
            and offtarget / total > max_offtarget_fraction
        ):
            result.add(s)
    return result


def _find_unclaimed_vx_segment_ids(
    seg_ids: np.ndarray,
    counts: np.ndarray,
    seg_total_vx: dict[int, int],
    max_unclaimed_vx: int | None,
    max_unclaimed_fraction: float | None,
) -> set[int]:
    """Find segments with too many unclaimed voxels (reference == 0)."""
    seg_total_overlap: dict[int, int] = defaultdict(int)
    for seg, cnt in zip(seg_ids, counts):
        seg_total_overlap[int(seg)] += int(cnt)

    result: set[int] = set()
    for s, total in seg_total_vx.items():
        unclaimed = total - seg_total_overlap.get(s, 0)
        if max_unclaimed_vx is not None and unclaimed > max_unclaimed_vx:
            result.add(s)
        elif (
            max_unclaimed_fraction is not None
            and total > 0
            and unclaimed / total > max_unclaimed_fraction
        ):
            result.add(s)
    return result


def _compute_dominant_labels(seg: np.ndarray, constraint: np.ndarray) -> dict[int, int]:
    """For each segment, find the constraint label covering >50% of its voxels.

    Returns dict mapping segment_id -> dominant label, only for segments
    where a single non-zero constraint label covers more than half of all voxels.
    """
    flat_seg = seg.ravel()
    flat_con = constraint.ravel()

    seg_mask = flat_seg != 0
    seg_ids_arr, seg_counts = np.unique(flat_seg[seg_mask], return_counts=True)
    seg_totals = dict(zip(seg_ids_arr.astype(int), seg_counts.astype(int)))

    valid_mask = seg_mask & (flat_con != 0)
    if not valid_mask.any():
        return {}

    df = pd.DataFrame({"seg": flat_seg[valid_mask], "con": flat_con[valid_mask]})
    pair_counts = df.groupby(["seg", "con"]).size().reset_index(name="count")

    idx_max = pair_counts.groupby("seg")["count"].idxmax()
    best = pair_counts.loc[idx_max]

    result: dict[int, int] = {}
    seg_arr = best["seg"].values
    con_arr = best["con"].values
    count_arr = best["count"].values
    for i in range(len(best)):
        seg_id = int(seg_arr[i])
        count = int(count_arr[i])
        total = seg_totals.get(seg_id, 0)
        if total > 0 and count > total / 2:
            result[seg_id] = int(con_arr[i])
    return result


def _compute_segment_metrics(seg_ids, ref_ids, counts, seg_total_vx, min_overlap_vx):
    """Compute raw per-segment filter metric values.

    Returns dict mapping segment_id -> dict of metric values.
    """
    seg_best: dict[int, int] = defaultdict(int)
    seg_total_overlap: dict[int, int] = defaultdict(int)
    for seg, ref, cnt in zip(seg_ids, ref_ids, counts):
        s, c = int(seg), int(cnt)
        seg_best[s] = max(seg_best[s], c)
        seg_total_overlap[s] += c

    result = {}
    for s, total in seg_total_vx.items():
        best = seg_best.get(s, 0)
        covered = seg_total_overlap.get(s, 0)
        unclaimed = total - covered
        offtarget = total - best
        result[s] = {
            "size_vx": total,
            "best_overlap_vx": best,
            "total_overlap_vx": covered,
            "offtarget_vx": offtarget,
            "offtarget_fraction": offtarget / total if total > 0 else 0.0,
            "unclaimed_vx": unclaimed,
            "unclaimed_fraction": unclaimed / total if total > 0 else 0.0,
        }
    return result


def _compute_contact_filter_stats(
    seg_a, seg_b, aff, x, y, z, reference, start, resolution, max_faces_per_contact=100
):
    """Compute per-contact filter metrics for stats collection.

    Returns DataFrame with one row per unique (seg_a, seg_b) pair.
    """
    sx, sy, sz = float(start[0]), float(start[1]), float(start[2])

    # Interface GT fraction (same logic as _filter_pairs_by_interface_gt)
    lo_x = np.floor(x - sx).astype(np.intp)
    lo_y = np.floor(y - sy).astype(np.intp)
    lo_z = np.floor(z - sz).astype(np.intp)
    hi_x = np.ceil(x - sx).astype(np.intp)
    hi_y = np.ceil(y - sy).astype(np.intp)
    hi_z = np.ceil(z - sz).astype(np.intp)
    ref_lo = reference[lo_x, lo_y, lo_z]
    ref_hi = reference[hi_x, hi_y, hi_z]
    both_have_gt = (ref_lo != 0) & (ref_hi != 0)

    df = pd.DataFrame(
        {
            "a": seg_a,
            "b": seg_b,
            "aff": aff,
            "x": x,
            "y": y,
            "z": z,
            "gt": both_have_gt,
        }
    )

    grouped = df.groupby(["a", "b"]).agg(
        contact_count=("a", "size"),
        mean_affinity=("aff", "mean"),
        interface_gt_fraction=("gt", "mean"),
        wx=("x", lambda v: (v * df.loc[v.index, "aff"]).sum()),
        wy=("y", lambda v: (v * df.loc[v.index, "aff"]).sum()),
        wz=("z", lambda v: (v * df.loc[v.index, "aff"]).sum()),
        aff_sum=("aff", "sum"),
        mean_x=("x", "mean"),
        mean_y=("y", "mean"),
        mean_z=("z", "mean"),
    )

    # COM: weighted if aff_sum > 0, else unweighted mean
    has_aff = grouped["aff_sum"].values > 0
    grouped["com_x"] = np.where(
        has_aff, grouped["wx"].values / grouped["aff_sum"].values, grouped["mean_x"].values
    )
    grouped["com_y"] = np.where(
        has_aff, grouped["wy"].values / grouped["aff_sum"].values, grouped["mean_y"].values
    )
    grouped["com_z"] = np.where(
        has_aff, grouped["wz"].values / grouped["aff_sum"].values, grouped["mean_z"].values
    )

    # Convert COM from voxel coords to nm
    grouped["com_x"] = grouped["com_x"] * resolution[0]
    grouped["com_y"] = grouped["com_y"] * resolution[1]
    grouped["com_z"] = grouped["com_z"] * resolution[2]

    # Collect per-pair contact faces as JSON (x_nm, y_nm, z_nm, aff), capped
    rx, ry, rz = float(resolution[0]), float(resolution[1]), float(resolution[2])
    faces_json = {}
    for (a, b), group in df.groupby(["a", "b"]):
        g = (
            group
            if len(group) <= max_faces_per_contact
            else group.sample(max_faces_per_contact, random_state=42)
        )
        faces = [
            [
                round(r["x"] * rx, 1),
                round(r["y"] * ry, 1),
                round(r["z"] * rz, 1),
                round(float(r["aff"]), 4),
            ]
            for _, r in g.iterrows()
        ]
        faces_json[(a, b)] = json.dumps(faces)
    grouped["contact_faces_nm"] = [faces_json[(a, b)] for a, b in grouped.index]

    result = grouped[
        [
            "contact_count",
            "mean_affinity",
            "interface_gt_fraction",
            "com_x",
            "com_y",
            "com_z",
            "contact_faces_nm",
        ]
    ].reset_index()
    result.columns = [
        "seg_a",
        "seg_b",
        "contact_count",
        "mean_affinity",
        "interface_gt_fraction",
        "com_x",
        "com_y",
        "com_z",
        "contact_faces_nm",
    ]
    return result


def _build_seg_to_ref(
    seg_ids: np.ndarray, ref_ids: np.ndarray, counts: np.ndarray, min_overlap_vx: int
) -> dict[int, set[int]]:
    """Build mapping from segment to reference CCs it overlaps with."""
    result: dict[int, set[int]] = defaultdict(set)
    for seg, ref, cnt in zip(seg_ids, ref_ids, counts):
        if cnt >= min_overlap_vx:
            result[int(seg)].add(int(ref))
    return result


def _compute_seg_to_ref_by_segment(
    seg: np.ndarray,
    reference: np.ndarray,
    min_overlap_vx: int,
    dominant_ref_only: bool = False,
) -> dict[int, set[int]]:
    """Build mapping from segment to reference segment IDs (not CCs) it overlaps with.

    This uses raw reference segment IDs, not connected components, so that
    merge decisions work correctly when a reference segment is non-contiguous
    within the chunk.

    When ``dominant_ref_only`` is True, each segment maps to a single-element set
    containing only the reference id with the largest overlap (still subject to
    ``min_overlap_vx``). This avoids false-positive merges from sliver overlaps
    when seg and reference come from different segmentation pipelines with
    misaligned supervoxel boundaries.
    """
    flat_seg = seg.ravel()
    flat_ref = reference.ravel()
    valid_mask = (flat_seg != 0) & (flat_ref != 0)
    df = pd.DataFrame({"seg": flat_seg[valid_mask], "ref": flat_ref[valid_mask]})
    counts_df = df.groupby(["seg", "ref"]).size().reset_index(name="count")
    counts_df = counts_df[counts_df["count"] >= min_overlap_vx]
    if counts_df.empty:
        return {}

    if dominant_ref_only:
        idx = counts_df.groupby("seg")["count"].idxmax()
        counts_df = counts_df.loc[idx]

    result: dict[int, set[int]] = defaultdict(set)
    for s, r in zip(counts_df["seg"].values, counts_df["ref"].values):
        result[int(s)].add(int(r))
    return result


def _blackout_segments(seg: np.ndarray, ids_to_remove: set[int]) -> np.ndarray:
    """Set specified segment IDs to 0."""
    if not ids_to_remove:
        return seg
    seg = seg.copy()
    mask = np.isin(seg, np.array(list(ids_to_remove), dtype=seg.dtype))
    seg[mask] = 0
    return seg


def _find_axis_contacts(
    seg_lo: np.ndarray,
    seg_hi: np.ndarray,
    aff_slice: np.ndarray,
    offset: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find contacts along one axis. Returns face centers."""
    mask = (seg_lo != seg_hi) & (seg_lo != 0) & (seg_hi != 0)
    idx = np.nonzero(mask)
    if len(idx[0]) == 0:
        empty_i, empty_f = np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        return empty_i, empty_i, empty_f, empty_f, empty_f, empty_f
    return (
        seg_lo[mask],
        seg_hi[mask],
        aff_slice[mask],
        idx[0].astype(np.float32) + offset[0],
        idx[1].astype(np.float32) + offset[1],
        idx[2].astype(np.float32) + offset[2],
    )


def _find_contacts(
    seg: np.ndarray, aff: np.ndarray, start: Vec3D
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find contacts between segments using affinity data."""
    sx, sy, sz = float(start[0]), float(start[1]), float(start[2])
    results = []

    for seg_lo, seg_hi, aff_slice, offset in [
        (seg[:-1], seg[1:], aff[0, 1:], (sx + 0.5, sy, sz)),
        (seg[:, :-1], seg[:, 1:], aff[1, :, 1:], (sx, sy + 0.5, sz)),
        (seg[:, :, :-1], seg[:, :, 1:], aff[2, :, :, 1:], (sx, sy, sz + 0.5)),
    ]:
        r = _find_axis_contacts(seg_lo, seg_hi, aff_slice, offset)
        if len(r[0]) > 0:
            results.append(r)

    if not results:
        empty_i, empty_f = np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        return empty_i, empty_i, empty_f, empty_f, empty_f, empty_f

    seg_a = np.concatenate([r[0] for r in results])
    seg_b = np.concatenate([r[1] for r in results])
    aff_vals = np.concatenate([r[2] for r in results])
    x = np.concatenate([r[3] for r in results])
    y = np.concatenate([r[4] for r in results])
    z = np.concatenate([r[5] for r in results])

    swap = seg_a > seg_b
    seg_a, seg_b = np.where(swap, seg_b, seg_a), np.where(swap, seg_a, seg_b)

    return seg_a, seg_b, aff_vals, x, y, z


def _keep_mask_for_pairs(
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    pair_keep: np.ndarray,
    pair_index: pd.MultiIndex,
) -> np.ndarray:
    """Map per-pair boolean to per-face boolean mask via pandas index lookup."""
    keep_series = pd.Series(pair_keep, index=pair_index)
    face_pairs = pd.MultiIndex.from_arrays([seg_a, seg_b])
    return keep_series.reindex(face_pairs, fill_value=False).values


def _filter_pairs_touching_boundary(
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    aff: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    start: Vec3D,
    shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exclude pairs that have any contact touching the padded boundary."""
    padded_start = np.array([start[0], start[1], start[2]])
    padded_end = padded_start + np.array(shape)
    on_boundary = (
        (x <= padded_start[0])
        | (x >= padded_end[0] - 1)
        | (y <= padded_start[1])
        | (y >= padded_end[1] - 1)
        | (z <= padded_start[2])
        | (z >= padded_end[2] - 1)
    )
    # Find boundary pairs using pandas groupby (vectorized)
    df = pd.DataFrame({"a": seg_a, "b": seg_b, "on_b": on_boundary})
    pair_has_boundary = df.groupby(["a", "b"])["on_b"].any()
    keep_pairs = ~pair_has_boundary.values
    keep = _keep_mask_for_pairs(seg_a, seg_b, keep_pairs, pair_has_boundary.index)
    return seg_a[keep], seg_b[keep], aff[keep], x[keep], y[keep], z[keep]


def _filter_pairs_by_com(
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    aff: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    start: Vec3D,
    shape: tuple[int, ...],
    crop_pad: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exclude pairs whose affinity-weighted COM falls outside the kernel region."""
    kernel_start = np.array([start[0], start[1], start[2]]) + np.array(crop_pad)
    kernel_end = np.array([start[0], start[1], start[2]]) + np.array(shape) - np.array(crop_pad)

    # Vectorized COM computation using pandas groupby
    df = pd.DataFrame({"a": seg_a, "b": seg_b, "x": x, "y": y, "z": z, "aff": aff})
    df["wx"] = df["x"] * df["aff"]
    df["wy"] = df["y"] * df["aff"]
    df["wz"] = df["z"] * df["aff"]

    grouped = df.groupby(["a", "b"]).agg(
        {
            "wx": "sum",
            "wy": "sum",
            "wz": "sum",
            "aff": "sum",
            "x": "mean",
            "y": "mean",
            "z": "mean",
        }
    )

    # Compute COM: weighted if aff_sum > 0, else unweighted mean
    aff_sum = grouped["aff"].values
    has_aff = aff_sum > 0
    com_x = np.where(has_aff, grouped["wx"].values / aff_sum, grouped["x"].values)
    com_y = np.where(has_aff, grouped["wy"].values / aff_sum, grouped["y"].values)
    com_z = np.where(has_aff, grouped["wz"].values / aff_sum, grouped["z"].values)

    # Check which pairs are inside kernel
    inside = (
        (com_x >= kernel_start[0])
        & (com_x < kernel_end[0])
        & (com_y >= kernel_start[1])
        & (com_y < kernel_end[1])
        & (com_z >= kernel_start[2])
        & (com_z < kernel_end[2])
    )

    # Vectorized: map pair-level inside boolean to per-face mask
    keep = _keep_mask_for_pairs(seg_a, seg_b, inside, grouped.index)
    return seg_a[keep], seg_b[keep], aff[keep], x[keep], y[keep], z[keep]


def _filter_pairs_by_interface_gt(
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    aff: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    reference: np.ndarray,
    start: Vec3D,
    min_interface_gt_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exclude pairs where GT coverage at the contact interface is below threshold.

    For each contact face, checks whether reference is non-zero on both adjacent voxels.
    Pairs where the fraction of faces with GT on both sides < threshold are excluded.
    """
    sx, sy, sz = float(start[0]), float(start[1]), float(start[2])
    lo_x = np.floor(x - sx).astype(np.intp)
    lo_y = np.floor(y - sy).astype(np.intp)
    lo_z = np.floor(z - sz).astype(np.intp)
    hi_x = np.ceil(x - sx).astype(np.intp)
    hi_y = np.ceil(y - sy).astype(np.intp)
    hi_z = np.ceil(z - sz).astype(np.intp)

    ref_lo = reference[lo_x, lo_y, lo_z]
    ref_hi = reference[hi_x, hi_y, hi_z]
    both_have_gt = (ref_lo != 0) & (ref_hi != 0)

    df = pd.DataFrame({"a": seg_a, "b": seg_b, "gt": both_have_gt})
    grouped = df.groupby(["a", "b"])["gt"].mean()
    keep_pairs = grouped.values >= min_interface_gt_fraction
    keep = _keep_mask_for_pairs(seg_a, seg_b, keep_pairs, grouped.index)
    return seg_a[keep], seg_b[keep], aff[keep], x[keep], y[keep], z[keep]


def _compute_contact_counts(seg_a: np.ndarray, seg_b: np.ndarray) -> dict[tuple[int, int], int]:
    """Count contacts per segment pair."""
    df = pd.DataFrame({"a": seg_a, "b": seg_b})
    counts_df = df.groupby(["a", "b"]).size()
    return {(int(a), int(b)): int(cnt) for (a, b), cnt in counts_df.items()}


def _build_contact_lookup(
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    cz: np.ndarray,
    aff: np.ndarray,
) -> dict[tuple[int, int], list[tuple[float, float, float, float]]]:
    """Build lookup from segment pair to contact face centers."""
    data: dict[tuple[int, int], list[tuple[float, float, float, float]]] = defaultdict(list)
    for a, b, x, y, z, af in zip(seg_a, seg_b, cx, cy, cz, aff):
        data[(int(a), int(b))].append((float(x), float(y), float(z), float(af)))
    return data


def _filter_faces_by_mean_affinity(
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    aff: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    min_affinity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Early filter: drop all faces belonging to pairs with mean affinity below threshold."""
    if min_affinity <= 0.0:
        return seg_a, seg_b, aff, x, y, z
    df = pd.DataFrame({"a": seg_a, "b": seg_b, "aff": aff})
    pair_mean = df.groupby(["a", "b"])["aff"].mean()
    keep_pairs = pair_mean.values >= min_affinity
    keep = _keep_mask_for_pairs(seg_a, seg_b, keep_pairs, pair_mean.index)
    return seg_a[keep], seg_b[keep], aff[keep], x[keep], y[keep], z[keep]


def _filter_by_mean_affinity(
    contact_data: dict[tuple[int, int], list[tuple[float, float, float, float]]],
    min_affinity: float,
) -> dict[tuple[int, int], list[tuple[float, float, float, float]]]:
    """Filter out contact pairs whose mean affinity is below threshold."""
    if min_affinity <= 0.0:
        return contact_data
    result = {}
    for pair, contacts in contact_data.items():
        mean_aff = sum(c[3] for c in contacts) / len(contacts)
        if mean_aff >= min_affinity:
            result[pair] = contacts
    return result


def _mesh_nbytes(mesh: trimesh.Trimesh) -> int:
    """Estimate memory usage of a trimesh in bytes."""
    return mesh.vertices.nbytes + mesh.faces.nbytes


def _fetch_mesh(
    cv: CloudVolume, seg_id: int, lod: int = 0
) -> tuple[int, trimesh.Trimesh | None, float]:
    """Returns (seg_id, mesh_or_none, elapsed_seconds)."""
    t0 = time.time()
    try:
        meshes = cv.mesh.get([seg_id], lod=lod, progress=False)
        mesh_obj = meshes.get(seg_id)
        if mesh_obj is None or len(mesh_obj.vertices) == 0 or len(mesh_obj.faces) == 0:
            return seg_id, None, time.time() - t0
        return (
            seg_id,
            trimesh.Trimesh(vertices=mesh_obj.vertices, faces=mesh_obj.faces),
            time.time() - t0,
        )
    except (MeshDecodeError, MeshMissingError):
        return seg_id, None, time.time() - t0


class MeshLRUCache:
    """LRU cache for trimesh meshes with a memory budget."""

    def __init__(self, cv: CloudVolume, max_bytes: int, num_procs: int = 1, lod: int = 0):
        from cachetools import LRUCache

        self.cv = cv
        self.num_procs = num_procs
        self.lod = lod
        self._cache: LRUCache = LRUCache(maxsize=max_bytes, getsizeof=_mesh_nbytes)
        self._failed: set[int] = set()
        self._fetch_times: list[tuple[int, float, int]] = []  # (seg_id, seconds, nbytes)

    def get(self, seg_id: int) -> trimesh.Trimesh | None:
        if seg_id in self._failed:
            return None
        mesh = self._cache.get(seg_id)
        if mesh is not None:
            return mesh
        _, mesh, elapsed = _fetch_mesh(self.cv, seg_id, self.lod)
        if mesh is None:
            self._failed.add(seg_id)
            self._fetch_times.append((seg_id, elapsed, 0))
            return None
        self._fetch_times.append((seg_id, elapsed, _mesh_nbytes(mesh)))
        self._cache[seg_id] = mesh
        return mesh

    def prefetch(self, seg_ids: list[int]) -> None:
        to_fetch = [s for s in seg_ids if s not in self._cache and s not in self._failed]
        if not to_fetch:
            return
        n_before = len(self._cache)
        total_fetched_bytes = 0
        batch_size = max(self.num_procs, 1)
        for i in range(0, len(to_fetch), batch_size):
            batch = to_fetch[i : i + batch_size]
            if self.num_procs <= 1:
                results = [_fetch_mesh(self.cv, s, self.lod) for s in batch]
            else:
                lod = self.lod
                with ThreadPoolExecutor(max_workers=self.num_procs) as ex:
                    results = list(ex.map(lambda s: _fetch_mesh(self.cv, s, lod), batch))
            for seg_id, mesh, elapsed in results:
                if mesh is None:
                    self._failed.add(seg_id)
                    self._fetch_times.append((seg_id, elapsed, 0))
                else:
                    nbytes = _mesh_nbytes(mesh)
                    total_fetched_bytes += nbytes
                    self._cache[seg_id] = mesh
                    self._fetch_times.append((seg_id, elapsed, nbytes))
        if total_fetched_bytes > 0 and total_fetched_bytes > self._cache.maxsize:
            raise RuntimeError(
                f"Mesh prefetch ({total_fetched_bytes / 1024**2:.0f}MB) exceeds "
                f"cache limit ({self._cache.maxsize / 1024**2:.0f}MB). "
                f"Increase mesh_cache_bytes."
            )

    def write_timing_log(self, path: str = "/tmp/mesh_download_times.csv") -> None:
        with open(path, "w") as f:
            f.write("seg_id,seconds,nbytes\n")
            for seg_id, elapsed, nbytes in self._fetch_times:
                f.write(f"{seg_id},{elapsed:.4f},{nbytes}\n")


def _download_meshes(
    cv: CloudVolume, segment_ids: list[int], num_procs: int = 1
) -> dict[int, trimesh.Trimesh]:
    """Download meshes without clipping (clipping done per-contact to sphere)."""
    if not segment_ids:
        return {}

    result: dict[int, trimesh.Trimesh] = {}
    failed_ids: list[int] = []

    def _fetch_one(seg_id: int) -> tuple[int, trimesh.Trimesh | None, str | None]:
        try:
            meshes = cv.mesh.get([seg_id], progress=False)
            mesh_obj = meshes.get(seg_id)
            if mesh_obj is None or len(mesh_obj.vertices) == 0 or len(mesh_obj.faces) == 0:
                return seg_id, None, None
            return seg_id, trimesh.Trimesh(vertices=mesh_obj.vertices, faces=mesh_obj.faces), None
        except (MeshDecodeError, MeshMissingError):
            return seg_id, None, "decode_error"

    batch_size = max(num_procs, 1)
    for batch_start in range(0, len(segment_ids), batch_size):
        batch = segment_ids[batch_start : batch_start + batch_size]
        if num_procs <= 1:
            fetch_results = [_fetch_one(seg_id) for seg_id in batch]
        else:
            with ThreadPoolExecutor(max_workers=num_procs) as executor:
                fetch_results = list(executor.map(_fetch_one, batch))

        for seg_id, mesh, error in fetch_results:
            if error == "decode_error":
                failed_ids.append(seg_id)
            elif mesh is not None:
                result[seg_id] = mesh

    if failed_ids:
        print(
            f"Mesh download: {len(failed_ids)}/{len(segment_ids)} "
            f"({100*len(failed_ids)/len(segment_ids):.1f}%) failed with MeshDecodeError"
        )

    return result


def _sort_pairs_by_segment_frequency(
    valid_pairs: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Sort pairs so segments appearing in the most contacts come first.

    This maximizes mesh cache hits by processing high-frequency segments together.
    """
    seg_counts: dict[int, int] = defaultdict(int)
    for a, b in valid_pairs:
        seg_counts[a] += 1
        seg_counts[b] += 1

    return sorted(valid_pairs, key=lambda p: -(seg_counts[p[0]] + seg_counts[p[1]]))


def _select_components_near_contacts(
    components: list[trimesh.Trimesh],
    contact_points: np.ndarray,
    touch_threshold: float = 100.0,
) -> trimesh.Trimesh:
    """Select and merge components that are within touch_threshold of contact points."""
    touching = [
        c
        for c in components
        if len(c.vertices) > 0 and cdist(c.vertices, contact_points).min() <= touch_threshold
    ]
    if not touching:
        return max(components, key=lambda c: len(c.faces))
    if len(touching) == 1:
        return touching[0]
    return trimesh.util.concatenate(touching)


def _select_best_component(
    components: list[trimesh.Trimesh],
    contact_points: np.ndarray | None,
) -> trimesh.Trimesh | None:
    """Select best component(s) from a list based on contact points."""
    if len(components) == 0:
        return None
    if len(components) == 1:
        return components[0]
    if contact_points is None or len(contact_points) == 0:
        return max(components, key=lambda c: len(c.faces))
    return _select_components_near_contacts(components, contact_points)


def _crop_mesh_to_sphere(
    mesh: trimesh.Trimesh,
    center: np.ndarray,
    radius: float,
    contact_points: np.ndarray | None = None,
) -> trimesh.Trimesh | None:
    """Clip mesh to sphere, keeping only components within touch_threshold of contact points."""
    vertex_dists = np.linalg.norm(mesh.vertices - center, axis=1)
    vertex_inside = vertex_dists <= radius
    face_inside = vertex_inside[mesh.faces].all(axis=1)

    if not face_inside.any():
        return None

    submesh_result = mesh.submesh([face_inside], append=True)
    cropped = submesh_result[0] if isinstance(submesh_result, list) else submesh_result

    if cropped is None or len(cropped.faces) == 0:
        return None

    components = cropped.split(only_watertight=False)
    return _select_best_component(components, contact_points)


def _sample_mesh_points(mesh: trimesh.Trimesh | None, n: int) -> np.ndarray:
    """Sample N points from mesh surface, area-weighted."""
    if mesh is None or len(mesh.faces) == 0:
        return np.zeros((n, 3), dtype=np.float32)
    # Use deterministic seed based on mesh geometry
    seed = int(np.abs(mesh.vertices.sum() * 1000)) % (2 ** 31)
    rng = np.random.RandomState(seed)
    result = trimesh.sample.sample_surface(mesh, n, seed=rng)
    points = result[0]
    return points.astype(np.float32)


def _compute_affinity_weighted_com(
    contacts: list[tuple[float, float, float, float]], resolution: np.ndarray
) -> np.ndarray:
    """Compute affinity-weighted center of mass in nm."""
    x = np.array([c[0] for c in contacts])
    y = np.array([c[1] for c in contacts])
    z = np.array([c[2] for c in contacts])
    aff = np.array([c[3] for c in contacts])
    aff_sum = aff.sum()
    if aff_sum == 0:
        return np.array(
            [x.mean() * resolution[0], y.mean() * resolution[1], z.mean() * resolution[2]]
        )
    return np.array(
        [
            (x * aff).sum() / aff_sum * resolution[0],
            (y * aff).sum() / aff_sum * resolution[1],
            (z * aff).sum() / aff_sum * resolution[2],
        ]
    )


def _make_contact_faces_array(
    contacts: list[tuple[float, float, float, float]], resolution: np.ndarray
) -> np.ndarray:
    """Create contact faces array (N, 4) with x, y, z, affinity in nm."""
    x = np.array([c[0] for c in contacts]) * resolution[0]
    y = np.array([c[1] for c in contacts]) * resolution[1]
    z = np.array([c[2] for c in contacts]) * resolution[2]
    aff = np.array([c[3] for c in contacts])
    return np.stack([x, y, z, aff], axis=1).astype(np.float32)


def _build_voxel_spatial_hash(
    voxels: np.ndarray,
) -> dict[tuple[int, int, int], list[int]]:
    """Build spatial hash mapping voxel coordinates to point indices."""
    coord_to_indices: dict[tuple[int, int, int], list[int]] = {}
    for i in range(len(voxels)):
        key = (int(voxels[i, 0]), int(voxels[i, 1]), int(voxels[i, 2]))
        if key not in coord_to_indices:
            coord_to_indices[key] = []
        coord_to_indices[key].append(i)
    return coord_to_indices


def _get_unvisited_neighbors(
    voxel_coord: tuple[int, int, int],
    coord_to_indices: dict[tuple[int, int, int], list[int]],
    visited: np.ndarray,
) -> list[int]:
    """Get unvisited neighbor indices for a voxel using 6-connectivity."""
    offsets = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    neighbors = []
    cx, cy, cz = voxel_coord
    for dx, dy, dz in offsets:
        neighbor_key = (cx + dx, cy + dy, cz + dz)
        for idx in coord_to_indices.get(neighbor_key, []):
            if not visited[idx]:
                neighbors.append(idx)
    return neighbors


def _compute_contact_connected_components(
    contact_faces: np.ndarray,
    resolution: np.ndarray,
) -> list[np.ndarray]:
    """Compute connected components of contact faces using 6-connectivity."""
    if len(contact_faces) == 0:
        return []

    voxels = np.round(contact_faces[:, :3] / resolution).astype(np.int64)
    coord_to_indices = _build_voxel_spatial_hash(voxels)

    visited = np.zeros(len(voxels), dtype=bool)
    components = []

    for start in range(len(voxels)):
        if visited[start]:
            continue

        component = [start]
        visited[start] = True
        queue = [start]

        while queue:
            current = queue.pop(0)
            voxel_coord = (
                int(voxels[current, 0]),
                int(voxels[current, 1]),
                int(voxels[current, 2]),
            )
            for neighbor_idx in _get_unvisited_neighbors(voxel_coord, coord_to_indices, visited):
                visited[neighbor_idx] = True
                component.append(neighbor_idx)
                queue.append(neighbor_idx)

        components.append(np.array(component))

    return components


def _find_contact_center(contact_faces: np.ndarray, resolution: np.ndarray) -> np.ndarray:
    """Find contact center as the point closest to mean of largest component."""
    xyz = contact_faces[:, :3]
    components = _compute_contact_connected_components(contact_faces, resolution)
    if components:
        xyz = xyz[max(components, key=len)]
    mean_pos = xyz.mean(axis=0)
    return xyz[np.argmin(np.linalg.norm(xyz - mean_pos, axis=1))]


def _sample_sphere_voxels(
    center_nm: np.ndarray,
    radius: float,
    seg_volume: np.ndarray,
    seg_start_nm: np.ndarray,
    resolution: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample voxels within sphere, return local indices and nm coordinates."""
    center_vx = np.round(center_nm / resolution).astype(np.int32)
    seg_start_vx = (seg_start_nm / resolution).astype(np.int32)
    pad_vx = (radius / resolution + 1).astype(np.int32)

    ranges = [
        np.arange(
            max(0, center_vx[i] - pad_vx[i] - seg_start_vx[i]),
            min(seg_volume.shape[i], center_vx[i] + pad_vx[i] + 1 - seg_start_vx[i]),
            dtype=np.int32,
        )
        for i in range(3)
    ]
    if any(len(r) == 0 for r in ranges):
        return np.array([]), np.array([]), np.array([]), np.array([])

    grids = np.meshgrid(*ranges, indexing="ij")
    local_vx = np.column_stack([g.ravel() for g in grids])
    global_vx = local_vx + seg_start_vx
    voxel_nm = global_vx.astype(np.float32) * resolution

    dist_sq = np.sum((voxel_nm - center_nm) ** 2, axis=1)
    in_sphere = dist_sq <= radius * radius

    return (
        local_vx[in_sphere],
        voxel_nm[in_sphere],
        seg_volume[local_vx[in_sphere, 0], local_vx[in_sphere, 1], local_vx[in_sphere, 2]],
        in_sphere,
    )


def _voxel_closest_to_mean(voxel_nm: np.ndarray, mask: np.ndarray) -> Vec3D[float] | None:
    """Find voxel closest to mean of masked voxels, return as Vec3D."""
    if not mask.any():
        return None
    pts = voxel_nm[mask]
    mean = pts.mean(axis=0)
    idx = np.argmin(np.sum((pts - mean) ** 2, axis=1))
    return Vec3D(float(pts[idx, 0]), float(pts[idx, 1]), float(pts[idx, 2]))


def _compute_representative_points(
    contact_faces: np.ndarray,
    seg_a_id: int,
    seg_b_id: int,
    seg_volume: np.ndarray,
    seg_start_nm: np.ndarray,
    resolution: np.ndarray,
) -> dict[int, Vec3D[float]]:
    """Compute representative points as voxels closest to segment centers within sphere."""
    contact_center = _find_contact_center(contact_faces, resolution)
    fallback = Vec3D(float(contact_center[0]), float(contact_center[1]), float(contact_center[2]))

    local_vx, voxel_nm, seg_ids, _ = _sample_sphere_voxels(
        contact_center, 200.0, seg_volume, seg_start_nm, resolution
    )
    if len(local_vx) == 0:
        return {seg_a_id: fallback, seg_b_id: fallback}

    pt_a = _voxel_closest_to_mean(voxel_nm, seg_ids == seg_a_id)
    pt_b = _voxel_closest_to_mean(voxel_nm, seg_ids == seg_b_id)

    return {seg_a_id: pt_a or fallback, seg_b_id: pt_b or fallback}


def _generate_seg_contact(
    contact_id: int,
    seg_a_id: int,
    seg_b_id: int,
    meshes: dict[int, trimesh.Trimesh],
    contact_data: dict[tuple[int, int], list[tuple[float, float, float, float]]],
    seg_to_ref: dict[int, set[int]],
    resolution: np.ndarray,
    pointcloud_configs: list[tuple[float, int]],
    merge_authority: str,
    seg_volume: np.ndarray,
    seg_start_nm: np.ndarray,
) -> SegContact | None:
    """Generate a single SegContact for a contact pair."""
    mesh_a, mesh_b = meshes.get(seg_a_id), meshes.get(seg_b_id)
    if mesh_a is None or mesh_b is None:
        return None

    com = _compute_affinity_weighted_com(contact_data[(seg_a_id, seg_b_id)], resolution)
    contact_faces = _make_contact_faces_array(contact_data[(seg_a_id, seg_b_id)], resolution)

    local_pointclouds = _generate_pointclouds(
        mesh_a, mesh_b, seg_a_id, seg_b_id, com, contact_faces[:, :3], pointcloud_configs
    )
    if not local_pointclouds:
        return None

    merge_decisions: dict[str, bool] = {}
    if seg_to_ref:
        merge_decisions = {
            merge_authority: bool(
                seg_to_ref.get(seg_a_id, set()) & seg_to_ref.get(seg_b_id, set())
            )
        }

    return SegContact(
        id=contact_id,
        seg_a=seg_a_id,
        seg_b=seg_b_id,
        com=Vec3D(float(com[0]), float(com[1]), float(com[2])),
        contact_faces=contact_faces,
        local_pointclouds=local_pointclouds,
        merge_decisions=merge_decisions,
        representative_points=_compute_representative_points(
            contact_faces, seg_a_id, seg_b_id, seg_volume, seg_start_nm, resolution
        ),
    )


def _generate_pointclouds(
    mesh_a: trimesh.Trimesh,
    mesh_b: trimesh.Trimesh,
    seg_a_id: int,
    seg_b_id: int,
    com: np.ndarray,
    contact_points_xyz: np.ndarray,
    pointcloud_configs: list[tuple[float, int]],
) -> dict[tuple[int, int], dict[int, np.ndarray]]:
    """Generate pointclouds for all configs."""
    result: dict[tuple[int, int], dict[int, np.ndarray]] = {}
    for radius_nm, n_points in pointcloud_configs:
        mesh_a_cropped = _crop_mesh_to_sphere(mesh_a, com, radius_nm, contact_points_xyz)
        mesh_b_cropped = _crop_mesh_to_sphere(mesh_b, com, radius_nm, contact_points_xyz)
        if mesh_a_cropped is None or mesh_b_cropped is None:
            continue
        result[(int(radius_nm), n_points)] = {
            seg_a_id: _sample_mesh_points(mesh_a_cropped, n_points),
            seg_b_id: _sample_mesh_points(mesh_b_cropped, n_points),
        }
    return result


@builder.register("SegContactOp")
@taskable_operation_cls
@attrs.frozen
class SegContactOp:
    """Operation to find and write segment contacts with pointclouds and merge decisions."""

    crop_pad: Sequence[int] = (0, 0, 0)
    min_seg_size_vx: int = 2000
    min_overlap_vx: int = 1000
    min_contact_vx: int = 5
    max_contact_vx: int = 2048
    min_affinity: float = 0.0
    merge_authority: str = "reference_overlap"
    dominant_ref_only: bool = False
    ids_per_chunk: int = 10000
    max_offtarget_vx: int | None = None
    max_offtarget_fraction: float | None = None
    offtarget_includes_unclaimed: bool = False
    max_unclaimed_vx: int | None = None
    max_unclaimed_fraction: float | None = None
    min_interface_gt_fraction: float | None = None
    collect_filter_stats_only: bool = False
    min_nucleus_vx: int = 1
    skip_chunks_with_nucleus: bool = False
    num_procs: int = attrs.Factory(lambda: os.cpu_count() or 1)
    mesh_cache_bytes: int = 2 * 1024 ** 3  # 2GB default
    mesh_lod: int = 0

    def get_input_resolution(self, dst_resolution: Vec3D[float]) -> Vec3D[float]:
        return dst_resolution

    def with_added_crop_pad(self, crop_pad: Vec3D[int]) -> SegContactOp:
        return attrs.evolve(self, crop_pad=Vec3D[int](*self.crop_pad) + crop_pad)

    def __call__(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        self,
        idx: VolumetricIndex,
        dst: VolumetricSegContactLayer,
        segmentation_layer: VolumetricLayer,
        affinity_layer: VolumetricLayer,
        reference_layer: VolumetricLayer | None = None,
        nucleus_layer: VolumetricLayer | None = None,
        constraint_layers: dict[str, VolumetricLayer] | None = None,
    ) -> None:
        t_start = time.time()
        coord_str = f"{int(idx.start[0])}_{int(idx.start[1])}_{int(idx.start[2])}"

        # Skip if chunk already exists
        chunk_idx = dst.backend.com_to_chunk_idx(idx.bbox.start)
        chunk_path = dst.backend.get_chunk_path(chunk_idx)
        fs, fs_path = __import__("fsspec").core.url_to_fs(chunk_path)
        if fs.exists(fs_path):
            print(f"[{coord_str}] Chunk already exists, skipping", flush=True)
            return

        idx_padded = idx.padded(Vec3D[int](*self.crop_pad))
        resolution = np.array([idx.resolution[0], idx.resolution[1], idx.resolution[2]])

        # Read pointcloud configs from destination layer info file
        pointcloud_configs = dst.backend.get_pointcloud_configs()

        # Check for nucleus in chunk bbox
        chunk_has_nucleus = False
        if nucleus_layer is not None and (
            self.collect_filter_stats_only or self.skip_chunks_with_nucleus
        ):
            nuc_info = get_info(nucleus_layer.backend.path)
            nuc_res = min(nuc_info["scales"], key=lambda s: np.prod(s["resolution"]))["resolution"]
            nucleus_idx = VolumetricIndex(
                bbox=idx.bbox,
                resolution=Vec3D(*nuc_res),
            )
            nucleus_data = np.asarray(nucleus_layer[nucleus_idx]).squeeze()
            n_nucleus_vx = int(np.count_nonzero(nucleus_data))
            chunk_has_nucleus = n_nucleus_vx >= self.min_nucleus_vx
            print(
                f"[{coord_str}] Nucleus check: {n_nucleus_vx} non-zero vx "
                f"-> {'HAS NUCLEUS' if chunk_has_nucleus else 'no nucleus'}",
                flush=True,
            )
            if self.skip_chunks_with_nucleus and chunk_has_nucleus:
                print(f"[{coord_str}] Skipping chunk with nucleus", flush=True)
                return

        # Read all layers
        t0 = time.time()
        with semaphore("read"):
            if reference_layer is not None:
                seg, reference, aff = _read_layers_parallel(
                    segmentation_layer, reference_layer, affinity_layer, idx_padded
                )
            else:
                seg = np.asarray(segmentation_layer[idx_padded]).squeeze()
                aff = np.asarray(affinity_layer[idx_padded])
                reference = None
        print(f"[{coord_str}] Read layers: {time.time() - t0:.1f}s", flush=True)

        # Read constraint layers and compute dominant labels per segment
        dominant_labels: dict[str, dict[int, int]] = {}
        if constraint_layers:
            for name, layer in constraint_layers.items():
                con_data = np.asarray(layer[idx_padded]).squeeze()
                dominant_labels[name] = _compute_dominant_labels(seg, con_data)
                print(
                    f"[{coord_str}] Constraint '{name}': "
                    f"{len(dominant_labels[name])} segments with dominant label",
                    flush=True,
                )

        # Compute overlaps and find segments to exclude
        t0 = time.time()
        small_ids = _find_small_segment_ids(seg, self.min_seg_size_vx)
        seg_to_ref: dict[int, set[int]] = {}

        if reference is not None:
            overlap_seg, overlap_ref, overlap_count = _compute_overlaps(seg, reference)
            all_seg_ids = set(int(s) for s in np.unique(seg) if s != 0)
            merger_ids = _find_merger_segment_ids(
                overlap_seg, overlap_ref, overlap_count, self.min_overlap_vx
            )
            unclaimed_ids = _find_unclaimed_segment_ids(
                overlap_seg, overlap_count, self.min_overlap_vx, all_seg_ids
            )

            unique_segs, seg_counts = np.unique(seg, return_counts=True)
            seg_total_vx = {int(s): int(c) for s, c in zip(unique_segs, seg_counts) if s != 0}

            if self.collect_filter_stats_only:
                seg_metrics = _compute_segment_metrics(
                    overlap_seg, overlap_ref, overlap_count, seg_total_vx, self.min_overlap_vx
                )
                print(
                    f"[{coord_str}] Computed segment metrics for {len(seg_metrics)} segments",
                    flush=True,
                )

                # Find contacts WITHOUT blackout
                t0 = time.time()
                seg_a, seg_b, aff_vals, x, y, z = _find_contacts(seg, aff, idx_padded.start)
                if len(seg_a) == 0:
                    print(f"[{coord_str}] No contacts found, skipping", flush=True)
                    return
                print(
                    f"[{coord_str}] Find contacts (unfiltered): {time.time() - t0:.1f}s "
                    f"({len(seg_a)} face voxels)",
                    flush=True,
                )

                # Apply only structural filters (boundary + COM)
                seg_a, seg_b, aff_vals, x, y, z = _filter_pairs_touching_boundary(
                    seg_a, seg_b, aff_vals, x, y, z, idx_padded.start, seg.shape
                )
                if len(seg_a) == 0:
                    print(f"[{coord_str}] All pairs on boundary, skipping", flush=True)
                    return

                seg_a, seg_b, aff_vals, x, y, z = _filter_pairs_by_com(
                    seg_a, seg_b, aff_vals, x, y, z, idx_padded.start, seg.shape, self.crop_pad
                )
                if len(seg_a) == 0:
                    print(f"[{coord_str}] All pairs outside kernel, skipping", flush=True)
                    return

                # Compute per-contact metrics
                contact_stats = _compute_contact_filter_stats(
                    seg_a, seg_b, aff_vals, x, y, z, reference, idx_padded.start, resolution
                )

                # GT merge label: unknown if either segment lacks GT overlap.
                # Compute BOTH set-based and dominant-ref-based mappings so
                # downstream filter-stats analysis can pick either criterion
                # without re-running. The op's `dominant_ref_only` flag selects
                # which one becomes the canonical `gt_merge_label` column.
                seg_to_ref_set = _compute_seg_to_ref_by_segment(
                    seg,
                    reference,
                    self.min_overlap_vx,
                    dominant_ref_only=False,
                )
                seg_to_ref_dom = _compute_seg_to_ref_by_segment(
                    seg,
                    reference,
                    self.min_overlap_vx,
                    dominant_ref_only=True,
                )

                def _label_from(mapping):
                    def _fn(row):
                        refs_a = mapping.get(int(row["seg_a"]))
                        refs_b = mapping.get(int(row["seg_b"]))
                        if refs_a is None or refs_b is None:
                            return "unknown"
                        return "merge" if refs_a & refs_b else "no_merge"

                    return _fn

                contact_stats["gt_merge_label_set"] = contact_stats.apply(
                    _label_from(seg_to_ref_set), axis=1
                )
                contact_stats["gt_merge_label_dominant"] = contact_stats.apply(
                    _label_from(seg_to_ref_dom), axis=1
                )
                contact_stats["gt_merge_label"] = (
                    contact_stats["gt_merge_label_dominant"]
                    if self.dominant_ref_only
                    else contact_stats["gt_merge_label_set"]
                )
                contact_stats["gt_refs_a"] = contact_stats["seg_a"].map(
                    lambda s: json.dumps(sorted(seg_to_ref_set.get(int(s), set())))
                )
                contact_stats["gt_refs_b"] = contact_stats["seg_b"].map(
                    lambda s: json.dumps(sorted(seg_to_ref_set.get(int(s), set())))
                )
                # Build as pd.array(..., dtype="UInt64") directly. Going through
                # .map() with `int | None` values would coerce the Series to float64,
                # silently rounding IDs above 2^53 to the nearest representable double.
                contact_stats["gt_dominant_ref_a"] = pd.array(
                    [
                        next(iter(seg_to_ref_dom.get(int(s), set())), None)
                        for s in contact_stats["seg_a"]
                    ],
                    dtype="UInt64",
                )
                contact_stats["gt_dominant_ref_b"] = pd.array(
                    [
                        next(iter(seg_to_ref_dom.get(int(s), set())), None)
                        for s in contact_stats["seg_b"]
                    ],
                    dtype="UInt64",
                )

                # Join per-segment metrics for both seg_a and seg_b
                for prefix, col in [("seg_a_", "seg_a"), ("seg_b_", "seg_b")]:
                    for metric in [
                        "size_vx",
                        "best_overlap_vx",
                        "total_overlap_vx",
                        "offtarget_vx",
                        "offtarget_fraction",
                        "unclaimed_vx",
                        "unclaimed_fraction",
                    ]:
                        contact_stats[prefix + metric] = contact_stats[col].map(
                            lambda s, m=metric: seg_metrics.get(int(s), {}).get(m, 0)
                        )

                # Derived worst-case columns
                contact_stats["min_size_vx"] = contact_stats[
                    ["seg_a_size_vx", "seg_b_size_vx"]
                ].min(axis=1)
                contact_stats["max_offtarget_vx"] = contact_stats[
                    ["seg_a_offtarget_vx", "seg_b_offtarget_vx"]
                ].max(axis=1)
                contact_stats["max_offtarget_fraction"] = contact_stats[
                    ["seg_a_offtarget_fraction", "seg_b_offtarget_fraction"]
                ].max(axis=1)
                contact_stats["max_unclaimed_vx"] = contact_stats[
                    ["seg_a_unclaimed_vx", "seg_b_unclaimed_vx"]
                ].max(axis=1)
                contact_stats["max_unclaimed_fraction"] = contact_stats[
                    ["seg_a_unclaimed_fraction", "seg_b_unclaimed_fraction"]
                ].max(axis=1)
                contact_stats["min_best_overlap_vx"] = contact_stats[
                    ["seg_a_best_overlap_vx", "seg_b_best_overlap_vx"]
                ].min(axis=1)

                # Check mesh availability
                seg_ids_list = list(
                    set(
                        contact_stats["seg_a"].astype(int).tolist()
                        + contact_stats["seg_b"].astype(int).tolist()
                    )
                )
                t_mesh = time.time()
                mesh_cv = CloudVolume(
                    segmentation_layer.backend.name, use_https=True, progress=False
                )
                if mesh_cv.info.get("mesh") is None:
                    # No mesh field in the layer's info → no meshes to check.
                    # Skip the call (saves a GCS round-trip) and record has_mesh_* = False.
                    contact_stats["has_mesh_a"] = False
                    contact_stats["has_mesh_b"] = False
                    contact_stats["both_meshes"] = False
                    print(
                        f"[{coord_str}] Mesh check: segmentation layer has no mesh field; "
                        f"skipping (has_mesh_* = False) "
                        f"({time.time() - t_mesh:.1f}s)",
                        flush=True,
                    )
                else:
                    mesh_exists_results = mesh_cv.mesh.exists(seg_ids_list)
                    # CloudVolume mesh sources return either:
                    #   - list [manifest_or_None] aligned with input (sharded multi-LOD)
                    #   - dict {str(id): manifest_or_None}        (other source classes)
                    # Handle both. A bare-zip implementation silently misbehaved on dict
                    # returns (iterates keys → all non-None → every seg marked has-mesh).
                    if isinstance(mesh_exists_results, dict):
                        has_mesh = {
                            int(seg_id)
                            for seg_id, result in mesh_exists_results.items()
                            if result is not None
                        }
                    else:
                        has_mesh = {
                            seg_id
                            for seg_id, result in zip(seg_ids_list, mesh_exists_results)
                            if result is not None
                        }
                    contact_stats["has_mesh_a"] = contact_stats["seg_a"].astype(int).isin(has_mesh)
                    contact_stats["has_mesh_b"] = contact_stats["seg_b"].astype(int).isin(has_mesh)
                    contact_stats["both_meshes"] = (
                        contact_stats["has_mesh_a"] & contact_stats["has_mesh_b"]
                    )
                    n_both = contact_stats["both_meshes"].sum()
                    print(
                        f"[{coord_str}] Mesh check: {len(has_mesh)}/{len(seg_ids_list)} "
                        f"segments have meshes, {n_both}/{len(contact_stats)} pairs have both "
                        f"({time.time() - t_mesh:.1f}s)",
                        flush=True,
                    )

                contact_stats["chunk_coord"] = coord_str
                contact_stats["has_nucleus"] = chunk_has_nucleus

                # Write parquet
                stats_path = f"{dst.backend.path}/filter_stats/{coord_str}.parquet"
                contact_stats.to_parquet(stats_path, index=False)
                print(
                    f"[{coord_str}] Wrote {len(contact_stats)} contact stats to {stats_path} "
                    f"({time.time() - t_start:.1f}s total)",
                    flush=True,
                )
                return

            # Per-segment GT coverage filters
            offtarget_ids: set[int] = set()
            if self.max_offtarget_vx is not None or self.max_offtarget_fraction is not None:
                offtarget_ids = _find_offtarget_segment_ids(
                    overlap_seg,
                    overlap_ref,
                    overlap_count,
                    seg_total_vx,
                    self.max_offtarget_vx,
                    self.max_offtarget_fraction,
                    self.offtarget_includes_unclaimed,
                )

            unclaimed_vx_ids: set[int] = set()
            if self.max_unclaimed_vx is not None or self.max_unclaimed_fraction is not None:
                unclaimed_vx_ids = _find_unclaimed_vx_segment_ids(
                    overlap_seg,
                    overlap_count,
                    seg_total_vx,
                    self.max_unclaimed_vx,
                    self.max_unclaimed_fraction,
                )

            exclude_ids = small_ids | merger_ids | unclaimed_ids | offtarget_ids | unclaimed_vx_ids
            print(
                f"[{coord_str}] Overlaps/filtering: {time.time() - t0:.1f}s "
                f"(excl {len(small_ids)} small, {len(merger_ids)} merger, "
                f"{len(unclaimed_ids)} unclaimed, {len(offtarget_ids)} offtarget, "
                f"{len(unclaimed_vx_ids)} unclaimed_vx, total {len(exclude_ids)} excluded)",
                flush=True,
            )

            # Build seg_to_ref mapping for merge decisions
            seg_to_ref = _compute_seg_to_ref_by_segment(
                seg,
                reference,
                self.min_overlap_vx,
                dominant_ref_only=self.dominant_ref_only,
            )
        else:
            exclude_ids = small_ids
            print(
                f"[{coord_str}] Filtering (no reference): {time.time() - t0:.1f}s "
                f"(excl {len(small_ids)} small)",
                flush=True,
            )

        # Blackout excluded segments
        seg = _blackout_segments(seg, exclude_ids)

        # Find contacts
        t0 = time.time()
        seg_a, seg_b, aff_vals, x, y, z = _find_contacts(seg, aff, idx_padded.start)
        if len(seg_a) == 0:
            print(f"[{coord_str}] No contacts found, skipping", flush=True)
            return
        print(
            f"[{coord_str}] Find contacts: {time.time() - t0:.1f}s ({len(seg_a)} voxels)",
            flush=True,
        )

        # Early affinity filter to reduce face count before expensive operations
        if self.min_affinity > 0.0:
            n_before = len(seg_a)
            seg_a, seg_b, aff_vals, x, y, z = _filter_faces_by_mean_affinity(
                seg_a, seg_b, aff_vals, x, y, z, self.min_affinity
            )
            if len(seg_a) == 0:
                print(f"[{coord_str}] No faces after early affinity filter, skipping", flush=True)
                return
            if len(seg_a) < n_before:
                print(
                    f"[{coord_str}] Early affinity filter: {n_before} -> {len(seg_a)} faces",
                    flush=True,
                )

        # Filter out pairs touching padded boundary (may have incomplete contacts)
        seg_a, seg_b, aff_vals, x, y, z = _filter_pairs_touching_boundary(
            seg_a, seg_b, aff_vals, x, y, z, idx_padded.start, seg.shape
        )
        if len(seg_a) == 0:
            print(f"[{coord_str}] All pairs on boundary, skipping", flush=True)
            return

        # Filter out pairs with COM outside kernel region
        seg_a, seg_b, aff_vals, x, y, z = _filter_pairs_by_com(
            seg_a, seg_b, aff_vals, x, y, z, idx_padded.start, seg.shape, self.crop_pad
        )
        if len(seg_a) == 0:
            print(f"[{coord_str}] All pairs outside kernel, skipping", flush=True)
            return

        # Filter out pairs with insufficient GT at the contact interface
        if self.min_interface_gt_fraction is not None and reference is not None:
            seg_a, seg_b, aff_vals, x, y, z = _filter_pairs_by_interface_gt(
                seg_a,
                seg_b,
                aff_vals,
                x,
                y,
                z,
                reference,
                idx_padded.start,
                self.min_interface_gt_fraction,
            )
            if len(seg_a) == 0:
                print(
                    f"[{coord_str}] All pairs below interface GT threshold, skipping",
                    flush=True,
                )
                return

        # Filter pairs by contact count
        contact_counts = _compute_contact_counts(seg_a, seg_b)
        valid_pairs: list[tuple[int, int]] = []
        segs_needing_mesh: set[int] = set()
        for (a, b), count in contact_counts.items():
            if self.min_contact_vx <= count <= self.max_contact_vx:
                valid_pairs.append((a, b))
                segs_needing_mesh.update([a, b])

        if not valid_pairs:
            print(f"[{coord_str}] No valid pairs after filtering, skipping", flush=True)
            return
        print(
            f"[{coord_str}] Valid pairs: {len(valid_pairs)}, "
            f"segs needing mesh: {len(segs_needing_mesh)}",
            flush=True,
        )

        # Build contact lookup
        contact_data = _build_contact_lookup(seg_a, seg_b, x, y, z, aff_vals)

        # Filter by mean affinity
        if self.min_affinity > 0.0:
            n_before = len(contact_data)
            contact_data = _filter_by_mean_affinity(contact_data, self.min_affinity)
            n_after = len(contact_data)
            if n_after < n_before:
                print(
                    f"[{coord_str}] Filtered by min_affinity={self.min_affinity}: "
                    f"{n_before} -> {n_after} pairs",
                    flush=True,
                )
            if not contact_data:
                print(f"[{coord_str}] No pairs after affinity filter, skipping", flush=True)
                return
            # Update valid_pairs to only include pairs that passed affinity filter
            valid_pairs = [(a, b) for (a, b) in valid_pairs if (a, b) in contact_data]
            segs_needing_mesh = set()
            for a, b in valid_pairs:
                segs_needing_mesh.update([a, b])

        # Filter by constraint layers (remove pairs where both have dominant labels that differ)
        if dominant_labels:
            for name, dom in dominant_labels.items():
                n_before = len(valid_pairs)
                valid_pairs = [
                    (a, b)
                    for a, b in valid_pairs
                    if not (a in dom and b in dom and dom[a] != dom[b])
                ]
                n_after = len(valid_pairs)
                if n_after < n_before:
                    print(
                        f"[{coord_str}] Constraint '{name}' filter: "
                        f"{n_before} -> {n_after} pairs",
                        flush=True,
                    )
            if not valid_pairs:
                print(
                    f"[{coord_str}] No pairs after constraint filters, skipping",
                    flush=True,
                )
                return
            segs_needing_mesh = set()
            for a, b in valid_pairs:
                segs_needing_mesh.update([a, b])

        # Sort pairs by segment frequency to maximize cache hits
        valid_pairs = _sort_pairs_by_segment_frequency(valid_pairs)

        # Download meshes and generate contacts using LRU cache
        id_offset = idx.chunk_id * self.ids_per_chunk
        seg_start_nm = np.array(
            [
                idx_padded.start[0] * resolution[0],
                idx_padded.start[1] * resolution[1],
                idx_padded.start[2] * resolution[2],
            ]
        )
        mesh_cv = CloudVolume(segmentation_layer.backend.name, use_https=True, progress=False)
        assert mesh_cv.info.get("mesh") is not None, (
            f"Segmentation layer {segmentation_layer.backend.name} has no 'mesh' field in "
            f"info. Contact generation requires meshes (run meshing first via igneous / "
            f"portal), or use collect_filter_stats_only=True."
        )
        mesh_cache = MeshLRUCache(
            cv=mesh_cv,
            max_bytes=self.mesh_cache_bytes,
            num_procs=self.num_procs,
            lod=self.mesh_lod,
        )
        contacts: list[SegContact] = []
        t_mesh_total, t_gen_total = 0.0, 0.0

        batch_size = max(self.num_procs, 1)
        for batch_start in range(0, len(valid_pairs), batch_size):
            batch_pairs = valid_pairs[batch_start : batch_start + batch_size]

            # Prefetch meshes for this batch in parallel
            batch_segs = []
            for a, b in batch_pairs:
                batch_segs.extend([a, b])
            t0 = time.time()
            mesh_cache.prefetch(list(set(batch_segs)))
            t_mesh_total += time.time() - t0

            # Generate contacts for this batch
            t0 = time.time()
            for local_id_offset, (seg_a_id, seg_b_id) in enumerate(batch_pairs):
                local_id = batch_start + local_id_offset
                mesh_a = mesh_cache.get(seg_a_id)
                mesh_b = mesh_cache.get(seg_b_id)
                if mesh_a is None or mesh_b is None:
                    continue
                meshes = {seg_a_id: mesh_a, seg_b_id: mesh_b}
                contact = _generate_seg_contact(
                    contact_id=id_offset + local_id,
                    seg_a_id=seg_a_id,
                    seg_b_id=seg_b_id,
                    meshes=meshes,
                    contact_data=contact_data,
                    seg_to_ref=seg_to_ref,
                    resolution=resolution,
                    pointcloud_configs=pointcloud_configs,
                    merge_authority=self.merge_authority,
                    seg_volume=seg,
                    seg_start_nm=seg_start_nm,
                )
                if contact is not None:
                    if dominant_labels:
                        meta = contact.partner_metadata or {}
                        for seg_id in (seg_a_id, seg_b_id):
                            seg_meta = meta.get(seg_id, {})
                            constraints = seg_meta.get("constraints", {})
                            for name, dom in dominant_labels.items():
                                if seg_id in dom:
                                    constraints[name] = dom[seg_id]
                            if constraints:
                                seg_meta["constraints"] = constraints
                            meta[seg_id] = seg_meta
                        contact.partner_metadata = meta
                    contacts.append(contact)
            t_gen_total += time.time() - t0

        mesh_cache.write_timing_log(f"/tmp/mesh_download_times_{coord_str}.csv")
        print(
            f"[{coord_str}] Download meshes: {t_mesh_total:.1f}s "
            f"(cached={len(mesh_cache._cache)}, "
            f"{mesh_cache._cache.currsize / 1024**2:.0f}MB, "
            f"failed={len(mesh_cache._failed)})",
            flush=True,
        )
        print(
            f"[{coord_str}] Generate contacts: {t_gen_total:.1f}s ({len(contacts)} contacts)",
            flush=True,
        )

        if contacts:
            if reference is not None:
                # Count merge decisions
                n_merge = sum(
                    1
                    for c in contacts
                    if c.merge_decisions and c.merge_decisions.get(self.merge_authority, False)
                )
                n_no_merge = len(contacts) - n_merge
                print(
                    f"[{coord_str}] Merge decisions: {n_merge} MERGE, {n_no_merge} NO-MERGE "
                    f"(total {len(contacts)})",
                    flush=True,
                )

            t0 = time.time()
            with semaphore("write"):
                dst[idx] = contacts
            print(f"[{coord_str}] Write: {time.time() - t0:.1f}s", flush=True)

        print(f"[{coord_str}] Total: {time.time() - t_start:.1f}s", flush=True)


@builder.register("AddPointcloudsOp")
@taskable_operation_cls
@attrs.frozen
class AddPointcloudsOp:
    """Add new pointcloud configs to existing contacts without recomputing contacts."""

    pointcloud_configs: Sequence[tuple[float, int]]  # [(radius_nm, n_points), ...]

    def get_input_resolution(self, dst_resolution: Vec3D[float]) -> Vec3D[float]:
        return dst_resolution

    def __call__(  # pylint: disable=too-many-locals
        self,
        idx: VolumetricIndex,
        dst: VolumetricSegContactLayer,
        segmentation_layer: VolumetricLayer,
    ) -> None:
        t_start = time.time()
        coord_str = f"{int(idx.start[0])}_{int(idx.start[1])}_{int(idx.start[2])}"

        # Convert idx to chunk_idx
        chunk_idx = dst.backend.com_to_chunk_idx(idx.bbox.start)

        # Read existing contacts (just contact data, no pointclouds needed)
        t0 = time.time()
        with semaphore("read"):
            contacts_data = dst.backend._read_contacts_chunk(chunk_idx)
        if not contacts_data:
            print(f"[{coord_str}] No contacts in chunk, skipping", flush=True)
            return
        print(
            f"[{coord_str}] Read contacts: {time.time() - t0:.1f}s "
            f"({len(contacts_data)} contacts)",
            flush=True,
        )

        # Get unique segment IDs
        segs_needing_mesh: set[int] = set()
        for c in contacts_data:
            segs_needing_mesh.add(c["seg_a"])
            segs_needing_mesh.add(c["seg_b"])

        # Download meshes
        t0 = time.time()
        mesh_cv = CloudVolume(segmentation_layer.backend.name, use_https=True, progress=False)
        assert mesh_cv.info.get("mesh") is not None, (
            f"Segmentation layer {segmentation_layer.backend.name} has no 'mesh' field in "
            f"info. Pointcloud generation requires meshes."
        )
        meshes = _download_meshes(mesh_cv, list(segs_needing_mesh))
        print(
            f"[{coord_str}] Download meshes: {time.time() - t0:.1f}s ({len(meshes)} meshes)",
            flush=True,
        )

        # Generate pointclouds for each config
        pointclouds_by_config: dict[
            tuple[int, int], list[tuple[int, int, int, np.ndarray, np.ndarray]]
        ] = {}

        t0 = time.time()
        for contact in contacts_data:
            contact_id = contact["id"]
            seg_a = contact["seg_a"]
            seg_b = contact["seg_b"]
            com = contact["com"]
            contact_faces = contact["contact_faces"]

            mesh_a = meshes.get(seg_a)
            mesh_b = meshes.get(seg_b)
            if mesh_a is None or mesh_b is None:
                continue

            com_np = np.array([com[0], com[1], com[2]])
            contact_points_xyz = contact_faces[:, :3] if contact_faces.shape[0] > 0 else None

            for radius_nm, n_points in self.pointcloud_configs:
                config_tuple = (int(radius_nm), n_points)

                mesh_a_cropped = _crop_mesh_to_sphere(
                    mesh_a, com_np, radius_nm, contact_points_xyz
                )
                mesh_b_cropped = _crop_mesh_to_sphere(
                    mesh_b, com_np, radius_nm, contact_points_xyz
                )
                if mesh_a_cropped is None or mesh_b_cropped is None:
                    continue

                pointcloud_a = _sample_mesh_points(mesh_a_cropped, n_points)
                pointcloud_b = _sample_mesh_points(mesh_b_cropped, n_points)

                if config_tuple not in pointclouds_by_config:
                    pointclouds_by_config[config_tuple] = []
                pointclouds_by_config[config_tuple].append(
                    (contact_id, seg_a, seg_b, pointcloud_a, pointcloud_b)
                )

        print(f"[{coord_str}] Generate pointclouds: {time.time() - t0:.1f}s", flush=True)

        # Write pointcloud chunks
        t0 = time.time()
        with semaphore("write"):
            for config_tuple, entries in pointclouds_by_config.items():
                dst.backend._write_pointcloud_chunk(chunk_idx, config_tuple, entries)
        print(
            f"[{coord_str}] Write pointclouds: {time.time() - t0:.1f}s "
            f"({len(pointclouds_by_config)} configs)",
            flush=True,
        )

        print(f"[{coord_str}] Total: {time.time() - t_start:.1f}s", flush=True)


@builder.register("ContactMergeOp")
@taskable_operation_cls
@attrs.frozen
class ContactMergeOp:
    """Operation to run contact merge inference and write merge_probabilities.

    Reads contacts from src layer, runs PointNet model inference,
    and writes updated contacts with merge_probabilities to dst layer.

    The conversion parameters must match the validation dataset configuration
    used during training to ensure consistent data representation.
    """

    model_path: str
    authority_name: str
    apply_sigmoid: bool = True
    include_contact_faces: bool = False
    contact_label: float | None = 0.0
    affinity_channel_mode: str | None = None
    config_key: tuple[int, int] | None = None
    max_batch_size: int | None = None
    use_constraints: bool = True
    csv_output_path: str | None = None

    def get_input_resolution(self, dst_resolution: Vec3D[float]) -> Vec3D[float]:
        return dst_resolution

    def with_added_crop_pad(self, crop_pad: Vec3D[int]) -> "ContactMergeOp":
        return self  # No crop pad needed for this op

    def __call__(
        self,
        idx: VolumetricIndex,
        dst: VolumetricSegContactLayer,
        src: VolumetricSegContactLayer,
    ) -> None:
        import torch

        from zetta_utils import convnet
        from zetta_utils.layer.volumetric.seg_contact.tensor_utils import (
            contacts_to_tensor,
        )

        t_start = time.time()
        coord_str = f"{int(idx.start[0])}_{int(idx.start[1])}_{int(idx.start[2])}"

        # Read contacts from source layer (with read procs for inference)
        t0 = time.time()
        with semaphore("read"):
            contacts = src[idx]
        if not contacts:
            logger.info(f"[{coord_str}] No contacts in chunk, skipping")
            return
        logger.info(
            f"[{coord_str}] Read contacts: {time.time() - t0:.1f}s ({len(contacts)} contacts)"
        )

        # Filter by constraints: skip contacts where both segments have a
        # non-zero dominant label for the same constraint key but they differ.
        # A value of 0 (no dominant label) is permissive — never causes filtering.
        if self.use_constraints:
            n_before = len(contacts)
            filtered = []
            for c in contacts:
                if c.partner_metadata:
                    con_a = c.partner_metadata.get(c.seg_a, {}).get("constraints", {})
                    con_b = c.partner_metadata.get(c.seg_b, {}).get("constraints", {})
                    shared_keys = set(con_a.keys()) & set(con_b.keys())
                    if any(
                        con_a[k] != 0 and con_b[k] != 0 and con_a[k] != con_b[k]
                        for k in shared_keys
                    ):
                        continue
                filtered.append(c)
            contacts = filtered
            if len(contacts) < n_before:
                logger.info(
                    f"[{coord_str}] Constraint filter: {n_before} -> {len(contacts)} contacts"
                )
            if not contacts:
                logger.info(f"[{coord_str}] No contacts after constraint filter, skipping")
                return

        n_contacts = len(contacts)

        # Convert to tensor for model input
        t0 = time.time()
        tensor, valid_indices = contacts_to_tensor(
            contacts,
            config_key=self.config_key,
            include_contact_faces=self.include_contact_faces,
            contact_label=self.contact_label,
            affinity_channel_mode=self.affinity_channel_mode,
        )
        logger.info(
            f"[{coord_str}] Convert to tensor: {time.time() - t0:.1f}s "
            f"({len(valid_indices)} valid contacts)"
        )

        if tensor.shape[0] == 0:
            logger.info(f"[{coord_str}] No valid contacts with pointclouds, skipping")
            return

        # Run model inference
        t0 = time.time()
        with semaphore("cuda"), torch.no_grad():
            if self.max_batch_size is None:
                output = convnet.utils.load_and_run_model(
                    path=self.model_path, data_in=tensor, autocast=False
                )
                if self.apply_sigmoid:
                    output = torch.sigmoid(output)
                probs = output.squeeze().cpu()
                torch.cuda.empty_cache()
            else:
                prob_chunks = []
                for i in range(0, tensor.shape[0], self.max_batch_size):
                    batch = tensor[i : i + self.max_batch_size]
                    out = convnet.utils.load_and_run_model(
                        path=self.model_path, data_in=batch, autocast=False
                    )
                    if self.apply_sigmoid:
                        out = torch.sigmoid(out)
                    prob_chunks.append(out.squeeze(-1).cpu())
                    torch.cuda.empty_cache()
                probs = torch.cat(prob_chunks, dim=0).squeeze()

        if probs.dim() == 0:
            probs = probs.unsqueeze(0)
        logger.info(f"[{coord_str}] Model inference: {time.time() - t0:.1f}s")

        # Build result with only IDs, COMs, and merge probabilities
        t0 = time.time()
        result = []
        for i, contact_idx in enumerate(valid_indices):
            contact = contacts[contact_idx]
            result.append(
                SegContact(
                    id=contact.id,
                    seg_a=contact.seg_a,
                    seg_b=contact.seg_b,
                    com=contact.com,
                    merge_probabilities={
                        self.authority_name: float(np.nan_to_num(probs[i].item(), nan=0.0))
                    },
                )
            )
        logger.info(f"[{coord_str}] Build results: {time.time() - t0:.1f}s")

        # Write CSV if configured
        if self.csv_output_path is not None:
            import csv as csv_module

            from cloudfiles import CloudFile

            rows = []
            for i, contact_idx in enumerate(valid_indices):
                contact = contacts[contact_idx]
                r = result[i]
                score = r.merge_probabilities.get(self.authority_name, "")
                mean_aff = ""
                if contact.contact_faces is not None and len(contact.contact_faces) > 0:
                    mean_aff = round(float(np.mean(contact.contact_faces[:, 3])), 3)
                n_faces = (
                    contact.contact_faces.shape[0] if contact.contact_faces is not None else 0
                )
                rows.append(
                    (
                        r.seg_a,
                        r.seg_b,
                        round(score, 3) if score != "" else "",
                        mean_aff,
                        n_faces,
                        int(r.com[0]),
                        int(r.com[1]),
                        int(r.com[2]),
                    )
                )
            rows.sort(key=lambda row: -float(row[2]) if row[2] != "" else 0)

            import io

            buf = io.StringIO()
            w = csv_module.writer(buf)
            w.writerow(
                [
                    "seg_a",
                    "seg_b",
                    "score",
                    "mean_affinity",
                    "n_faces",
                    "com_x_nm",
                    "com_y_nm",
                    "com_z_nm",
                ]
            )
            w.writerows(rows)
            csv_path = f"{self.csv_output_path}/{coord_str}.csv"
            CloudFile(csv_path).put(buf.getvalue().encode("utf-8"))
            logger.info(f"[{coord_str}] Wrote CSV: {csv_path} ({len(rows)} rows)")

        # Write only merge probabilities to destination
        t0 = time.time()
        with semaphore("write"):
            dst[idx] = result
        logger.info(f"[{coord_str}] Write merge probabilities: {time.time() - t0:.1f}s")

        logger.info(f"[{coord_str}] Total ContactMergeOp: {time.time() - t_start:.1f}s")


@builder.register("ContactEdgeFilterOp")
@taskable_operation_cls
@attrs.frozen
class ContactEdgeFilterOp:
    """Operation to filter seg contacts into .edges files by affinity and merge probability.

    Reads contacts from a VolumetricSegContactLayer and writes two pickled edge
    sets: one filtered by mean affinity threshold and one by merge probability
    threshold. Each edge is a (seg_a, seg_b, score) tuple.
    """

    affinity_thresholds: list[float]
    score_thresholds: list[float]
    authority_name: str

    def get_input_resolution(self, dst_resolution: Vec3D[float]) -> Vec3D[float]:
        return dst_resolution

    def with_added_crop_pad(self, crop_pad: Vec3D[int]) -> "ContactEdgeFilterOp":
        return self

    def __call__(
        self,
        idx: VolumetricIndex,
        dst: VolumetricSegContactLayer,
    ) -> None:
        import os
        import pickle

        from cloudfiles import CloudFile

        idx_str = idx.pformat()

        with semaphore("read"):
            contacts = dst[idx]

        if not contacts:
            logger.info(f"[{idx_str}] No contacts, skipping")
            return

        for aff_thr in self.affinity_thresholds:
            aff_edges: set[tuple] = set()
            for c in contacts:
                mean_aff = (
                    float(np.mean(c.contact_faces[:, 3]))
                    if c.contact_faces is not None and len(c.contact_faces) > 0
                    else 0.0
                )
                if mean_aff >= aff_thr:
                    aff_edges.add((np.uint64(c.seg_a), np.uint64(c.seg_b), mean_aff))
            aff_dir = os.path.join(dst.backend.name, "filtered_edges", f"aff_{aff_thr}")
            CloudFile(os.path.join(aff_dir, f"{idx_str}.edges")).put(pickle.dumps(aff_edges))
            logger.info(f"[{idx_str}] Wrote {len(aff_edges)} edges to aff_{aff_thr}/")

        for score_thr in self.score_thresholds:
            score_edges: set[tuple] = set()
            for c in contacts:
                if c.merge_probabilities and self.authority_name in c.merge_probabilities:
                    prob = c.merge_probabilities[self.authority_name]
                    if prob >= score_thr:
                        score_edges.add((np.uint64(c.seg_a), np.uint64(c.seg_b), prob))
            score_dir = os.path.join(dst.backend.name, "filtered_edges", f"score_{score_thr}")
            CloudFile(os.path.join(score_dir, f"{idx_str}.edges")).put(pickle.dumps(score_edges))
            logger.info(f"[{idx_str}] Wrote {len(score_edges)} edges to score_{score_thr}/")
