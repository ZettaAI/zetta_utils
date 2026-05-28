"""
Compare ground truth synapse line annotations to predicted assignments.

Reports two metrics, both derived from the same clustered view:
  1. DETECTION — one event per (pre_root + presyn-proximity) cluster.
  2. ASSIGNMENT — one sample per (pre_root, post_root) partner pair within
     matched detections.

For conventional (1-partner) synapses these reduce to per-synapse metrics;
for ribbons with multiple postsynaptic partners per presynaptic event the
detection metric counts ribbons and the assignment metric counts each partner.

Uses the ChunkedGraph (via CAVEclient) to resolve watershed supervoxel IDs to
proofread root IDs.  For GT points, the watershed layer is read to obtain SVs
first; for predictions, SV IDs come from the parquet metadata.

Outputs Neuroglancer local JSON annotation layers for TP/FP/FN visualization.

Usage:
    python scripts/synapse_evaluation/eval_synapses.py \
        --gt-json specs/nico/inference/cra9/synapses/gt024_gt_lines.json \
        --pred-metadata gs://dkronauer-ant-001-synapse/nkem/cutouts/2026-04-10/gt024_exp0328_50k/assignment/metadata \
        --watershed-path gs://zetta_ws/dkronauer-ant-001-240904-finetune-v3.2-0.27 \
        --cave-datastack kronauer_ant \
        --output-dir specs/nico/inference/cra9/synapses/eval_gt024/
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
from math import floor

import numpy as np
import pandas as pd
from caveclient import CAVEclient
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from zetta_utils.geometry import Vec3D
from zetta_utils.layer.volumetric.cloudvol import build_cv_layer

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def dim_to_nm(val_and_unit: list) -> float:
    value, unit = val_and_unit
    if unit == "m":
        value *= 1e9
    elif unit == "mm":
        value *= 1e6
    elif unit in ("um", "µm"):
        value *= 1e3
    elif unit != "nm":
        raise ValueError(f"Unknown unit: {val_and_unit}")
    return value


def load_gt(path: str) -> tuple[list[dict], Vec3D]:
    with open(path) as f:
        data = json.load(f)
    annotations = [a for a in data["annotations"] if a["type"] == "line"]
    transform = data["source"]["transform"]
    dims = transform.get("inputDimensions") or transform["outputDimensions"]
    if "0" in dims:
        res = Vec3D(dim_to_nm(dims["0"]), dim_to_nm(dims["1"]), dim_to_nm(dims["2"]))
    else:
        res = Vec3D(dim_to_nm(dims["x"]), dim_to_nm(dims["y"]), dim_to_nm(dims["z"]))
    return annotations, res


def gsutil_ls(path: str) -> list[str]:
    result = subprocess.run(["gsutil", "ls", path], check=True, capture_output=True, text=True)
    return result.stdout.strip().splitlines()


def gsutil_cp(src: str, dst: str) -> None:
    subprocess.run(["gsutil", "cp", src, dst], check=True, capture_output=True)


_CHUNK_FNAME_RE = re.compile(
    r"(?P<sx>-?\d+)-(?P<ex>-?\d+)"
    r"_(?P<sy>-?\d+)-(?P<ey>-?\d+)"
    r"_(?P<sz>-?\d+)-(?P<ez>-?\d+)\.parquet$"
)


def load_predictions(metadata_path: str) -> pd.DataFrame:
    """Load all rows of the tabular metadata layer at `metadata_path`.

    Filters out stale parquet files left over from prior runs at a different
    chunk_size: only files whose filename ranges match `info.chunk_size` and
    align to the dataset's chunk grid are loaded.
    """
    from zetta_utils.layer.volumetric.tabular.backend import read_info

    base = metadata_path.rstrip("/")
    info = read_info(base)
    cs = info["chunk_size"]
    vo = info["voxel_offset"]
    sz = info["size"]

    all_files = gsutil_ls(base + "/**")
    parquet_paths = [p for p in all_files if p.endswith(".parquet")]
    if not parquet_paths:
        raise RuntimeError(f"No parquet files found under {base}")

    valid_paths: list[str] = []
    skipped_stale = 0
    for p in parquet_paths:
        m = _CHUNK_FNAME_RE.search(p.rsplit("/", 1)[-1])
        if not m:
            skipped_stale += 1
            continue
        sx, ex = int(m["sx"]), int(m["ex"])
        sy, ey = int(m["sy"]), int(m["ey"])
        sz_, ez = int(m["sz"]), int(m["ez"])
        # Width must match info.chunk_size (last chunk along an axis can be
        # smaller if the dataset doesn't divide evenly, so allow that too).
        widths = (ex - sx, ey - sy, ez - sz_)
        starts = (sx, sy, sz_)
        ends = (ex, ey, ez)
        ok = True
        for d in range(3):
            ds_end = vo[d] + sz[d]  # dataset end on this axis
            full = widths[d] == cs[d]
            tail = ends[d] == ds_end and (ends[d] - starts[d]) <= cs[d]
            grid_aligned = (starts[d] - vo[d]) % cs[d] == 0
            if not grid_aligned or not (full or tail):
                ok = False
                break
        if ok:
            valid_paths.append(p)
        else:
            skipped_stale += 1
    if skipped_stale:
        print(f"  Skipped {skipped_stale} stale parquet file(s) not matching chunk_size={cs}")
    if not valid_paths:
        raise RuntimeError(
            f"No parquet files at chunk_size={cs} found under {base} "
            f"(skipped {skipped_stale} mismatched files)"
        )

    chunks = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for gcs_path in valid_paths:
            local = os.path.join(tmpdir, os.path.basename(gcs_path))
            gsutil_cp(gcs_path, local)
            chunks.append(pd.read_parquet(local))
    return pd.concat(chunks, ignore_index=True)


# ---------------------------------------------------------------------------
# Watershed SV lookup + ChunkedGraph root ID resolution
# ---------------------------------------------------------------------------


def lookup_sv_id(point_voxel: Vec3D, ws_data: np.ndarray, ws_start: Vec3D) -> int:
    local = Vec3D(*(floor(point_voxel[i] - ws_start[i]) for i in range(3)))
    try:
        return int(ws_data[local[0], local[1], local[2]])
    except (IndexError, ValueError):
        return 0


def load_watershed_region(watershed_path: str, all_points_voxel: list[Vec3D], resolution: Vec3D):
    pad = Vec3D(2, 2, 2)
    mins = Vec3D(*(min(p[i] for p in all_points_voxel) for i in range(3)))
    maxs = Vec3D(*(max(p[i] for p in all_points_voxel) for i in range(3)))
    start = Vec3D(*(floor(mins[i]) for i in range(3))) - pad
    end = Vec3D(*(floor(maxs[i]) + 1 for i in range(3))) + pad

    ws_layer = build_cv_layer(watershed_path, default_desired_resolution=resolution)
    data = ws_layer[resolution, start[0] : end[0], start[1] : end[1], start[2] : end[2]][0]
    return data, start


def resolve_roots(sv_ids: list[int], cave_client: "CAVEclient | None") -> list[int]:
    """Convert supervoxel IDs to root IDs via ChunkedGraph.  SV 0 maps to root 0.

    When *cave_client* is ``None`` (flat segmentation, no ChunkedGraph), the
    segment IDs are returned as-is — they already ARE the final IDs.
    """
    if not sv_ids:
        return []
    if cave_client is None:
        return sv_ids
    unique_svs = list({sv for sv in sv_ids if sv != 0})
    if not unique_svs:
        return [0] * len(sv_ids)

    print(f"  Resolving {len(unique_svs)} unique SVs to root IDs via ChunkedGraph...")
    root_map = {}
    batch_size = 500
    for i in range(0, len(unique_svs), batch_size):
        batch = unique_svs[i : i + batch_size]
        roots = cave_client.chunkedgraph.get_roots(batch)
        for sv, root in zip(batch, roots):
            root_map[sv] = int(root)

    return [root_map.get(sv, 0) for sv in sv_ids]


# ---------------------------------------------------------------------------
# Point matching
# ---------------------------------------------------------------------------


def match_points(
    gt_pts_nm: list[Vec3D],
    pred_pts_nm: list[Vec3D],
    max_dist_nm: float,
    valid_fn=None,
) -> dict:
    n_gt, n_pred = len(gt_pts_nm), len(pred_pts_nm)
    if n_gt == 0 or n_pred == 0:
        return {
            "matches": [],
            "tp": 0,
            "fp": n_pred,
            "fn": n_gt,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    kdtree = cKDTree(np.array(pred_pts_nm))
    dist_matrix = np.full((n_gt, n_pred), 1e9)

    for i, gp in enumerate(gt_pts_nm):
        dists, idxs = kdtree.query(np.array(gp), k=n_pred, distance_upper_bound=max_dist_nm)
        if n_pred == 1:
            dists, idxs = [dists], [idxs]
        for d, j in zip(dists, idxs):
            if j < n_pred and d < max_dist_nm:
                if valid_fn is None or valid_fn(i, j):
                    dist_matrix[i, j] = d

    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    matches = [
        (int(i), int(j)) for i, j in zip(row_ind, col_ind) if dist_matrix[i, j] < max_dist_nm
    ]

    tp = len(matches)
    fp = n_pred - tp
    fn = n_gt - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    matched_gt = {m[0] for m in matches}
    matched_pred = {m[1] for m in matches}

    result = {
        "matches": matches,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fp_pred_indices": [j for j in range(n_pred) if j not in matched_pred],
        "fn_gt_indices": [i for i in range(n_gt) if i not in matched_gt],
    }

    if matches:
        dists = sorted(dist_matrix[i, j] for i, j in matches)
        result["mean_dist"] = sum(dists) / len(dists)
        result["median_dist"] = dists[len(dists) // 2]
        result["max_dist"] = dists[-1]
        result["min_dist"] = dists[0]

    return result


# ---------------------------------------------------------------------------
# Key-based dedup (mutates a stats dict in place)
# ---------------------------------------------------------------------------


def _apply_key_dedup(
    stats: dict,
    gt_keys: list,
    pred_keys: list,
    gt_pts_nm: list,
    pred_pts_nm: list,
    max_dist_nm: float,
    boundary_excluded: "set[int] | None" = None,
) -> None:
    """Add dup_{tp,fp,fn} counters and {precision,recall,f1}_dedup to *stats*.

    Given a per-match *stats* dict from match_points() plus a hashable *key*
    per GT/pred (e.g. post_root, pre_root, or a (pre_root, post_root) tuple):

      - Unmatched pred with same key as a nearby TP → dup_tp (remove from FP).
      - Remaining unmatched preds sharing a key & nearby → dup_fp (keep one).
      - Unmatched GT whose key matches a nearby TP → dup_fn_matched (redundant
        label).
      - Remaining unmatched GTs sharing a key & nearby → dup_fn_unmatched
        (genuine miss, counted once).

    "Valid key" = non-zero scalar, or tuple with all non-zero components.
    """
    from collections import defaultdict

    boundary_excluded = boundary_excluded or set()

    def is_valid(k) -> bool:
        if isinstance(k, tuple):
            return all(x != 0 for x in k)
        return k != 0

    matches = stats["matches"]

    # --- Duplicate TPs (unmatched pred same key as nearby TP) ---
    tp_pred_keys: set = set()
    tp_pred_locs: list[tuple] = []
    for gi, pi in matches:
        k = pred_keys[pi]
        if is_valid(k):
            tp_pred_keys.add(k)
            tp_pred_locs.append((k, pred_pts_nm[pi]))

    dup_tp_indices: list[int] = []
    remaining_fp: list[int] = []
    for j in stats["fp_pred_indices"]:
        if j in boundary_excluded:
            continue
        k = pred_keys[j]
        if is_valid(k) and k in tp_pred_keys:
            pt = pred_pts_nm[j]
            if any(
                sum((pt[d] - loc[d]) ** 2 for d in range(3)) ** 0.5 < max_dist_nm
                for kk, loc in tp_pred_locs
                if kk == k
            ):
                dup_tp_indices.append(j)
                continue
        remaining_fp.append(j)

    # --- Duplicate FPs (remaining FPs sharing key & nearby) ---
    fp_by_key: dict = defaultdict(list)
    fp_no_key: list[int] = []
    for j in remaining_fp:
        k = pred_keys[j]
        (fp_by_key[k] if is_valid(k) else fp_no_key).append(j)

    dup_fp_indices: list[int] = []
    final_fp: list[int] = list(fp_no_key)
    for _, idxs in fp_by_key.items():
        if len(idxs) == 1:
            final_fp.append(idxs[0])
            continue
        kept = [idxs[0]]
        for idx in idxs[1:]:
            pt = pred_pts_nm[idx]
            if any(
                sum((pt[d] - pred_pts_nm[k][d]) ** 2 for d in range(3)) ** 0.5 < max_dist_nm
                for k in kept
            ):
                dup_fp_indices.append(idx)
            else:
                kept.append(idx)
        final_fp.extend(kept)

    # --- Duplicate FNs ---
    tp_gt_keys: set = set()
    tp_gt_locs: list[tuple] = []
    for gi, _ in matches:
        k = gt_keys[gi]
        if is_valid(k):
            tp_gt_keys.add(k)
            tp_gt_locs.append((k, gt_pts_nm[gi]))

    dup_fn_matched: list[int] = []
    dup_fn_unmatched: list[int] = []
    unresolved_fn: list[int] = []
    remaining_fn: list[int] = []
    for i in stats["fn_gt_indices"]:
        k = gt_keys[i]
        if not is_valid(k):
            remaining_fn.append(i)
            continue
        if k in tp_gt_keys:
            pt = gt_pts_nm[i]
            if any(
                sum((pt[d] - loc[d]) ** 2 for d in range(3)) ** 0.5 < max_dist_nm
                for kk, loc in tp_gt_locs
                if kk == k
            ):
                dup_fn_matched.append(i)
                continue
        unresolved_fn.append(i)

    fn_by_key: dict = defaultdict(list)
    for i in unresolved_fn:
        fn_by_key[gt_keys[i]].append(i)
    for _, idxs in fn_by_key.items():
        if len(idxs) == 1:
            remaining_fn.append(idxs[0])
            continue
        kept = [idxs[0]]
        for idx in idxs[1:]:
            pt = gt_pts_nm[idx]
            if any(
                sum((pt[d] - gt_pts_nm[k][d]) ** 2 for d in range(3)) ** 0.5 < max_dist_nm
                for k in kept
            ):
                dup_fn_unmatched.append(idx)
            else:
                kept.append(idx)
        remaining_fn.extend(kept)

    n_dup_tp = len(dup_tp_indices)
    n_dup_fp = len(dup_fp_indices)
    n_dup_fn_m = len(dup_fn_matched)
    n_dup_fn_u = len(dup_fn_unmatched)
    n_dup_fn = n_dup_fn_m + n_dup_fn_u

    tp_adj = stats["tp"]
    fp_adj = stats["fp"] - n_dup_tp - n_dup_fp
    fn_adj = stats["fn"] - n_dup_fn
    p_adj = tp_adj / (tp_adj + fp_adj) if (tp_adj + fp_adj) > 0 else 0.0
    r_adj = tp_adj / (tp_adj + fn_adj) if (tp_adj + fn_adj) > 0 else 0.0
    f1_adj = 2 * p_adj * r_adj / (p_adj + r_adj) if (p_adj + r_adj) > 0 else 0.0

    stats.update(
        {
            "dup_tp": n_dup_tp,
            "dup_fp": n_dup_fp,
            "dup_fn": n_dup_fn,
            "dup_fn_matched": n_dup_fn_m,
            "dup_fn_unmatched": n_dup_fn_u,
            "precision_dedup": p_adj,
            "recall_dedup": r_adj,
            "f1_dedup": f1_adj,
            "fp_pred_indices": final_fp,
            "dup_tp_pred_indices": dup_tp_indices,
            "dup_fp_pred_indices": dup_fp_indices,
            "fn_gt_indices": remaining_fn,
            "dup_fn_matched_gt_indices": dup_fn_matched,
            "dup_fn_unmatched_gt_indices": dup_fn_unmatched,
        }
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    gt_annotations: list[dict],
    pred_df: pd.DataFrame,
    watershed_path: str | None,
    cave_client: "CAVEclient | None",
    resolution: Vec3D,
    max_dist_nm: float,
    boundary_margin: tuple[int, int, int] | None = None,
    metadata_path: str | None = None,
    segmentation_path: str | None = None,
    synapse_type: str = "postsyn",
) -> tuple[dict, dict, dict]:
    # --- Read bbox from metadata info for filtering ---
    from zetta_utils.layer.volumetric.tabular.backend import read_info

    bbox_st = bbox_end = None
    if metadata_path is not None:
        info = read_info(metadata_path)
        voxel_offset = info["voxel_offset"]
        size = info["size"]
        bbox_st = Vec3D(*voxel_offset)
        bbox_end = bbox_st + Vec3D(*size)
        print(f"  Processing bbox: [{bbox_st}] -- [{bbox_end}]")

    # --- Filter GT and predictions to bbox ---
    gt_ignored: list[dict] = []
    if bbox_st is not None and bbox_end is not None:
        # Anchor side depends on synapse_type: presyn mode (vesicle/ribbon)
        # anchors on pointA, postsyn mode (PST/cleft) anchors on pointB.
        anchor_key = "pointA" if synapse_type == "presyn" else "pointB"
        gt_filtered = []
        for a in gt_annotations:
            pt = Vec3D(*a[anchor_key])
            if all(bbox_st[d] <= pt[d] < bbox_end[d] for d in range(3)):
                gt_filtered.append(a)
            else:
                gt_ignored.append(a)
        if gt_ignored:
            print(
                f"  Filtered GT to bbox ({anchor_key}): {len(gt_filtered)}/{len(gt_annotations)} ({len(gt_ignored)} ignored)"
            )
        gt_annotations = gt_filtered

        # Filter predictions: centroid must be inside bbox
        n_before = len(pred_df)
        mask = (
            (pred_df["centroid_x"] >= bbox_st[0])
            & (pred_df["centroid_x"] < bbox_end[0])
            & (pred_df["centroid_y"] >= bbox_st[1])
            & (pred_df["centroid_y"] < bbox_end[1])
            & (pred_df["centroid_z"] >= bbox_st[2])
            & (pred_df["centroid_z"] < bbox_end[2])
        )
        pred_df = pred_df[mask].reset_index(drop=True)
        n_outside = n_before - len(pred_df)
        if n_outside > 0:
            print(f"  Filtered {n_outside} predictions outside bbox")

    gt_postB = [Vec3D(*a["pointB"]) for a in gt_annotations]
    gt_preA = [Vec3D(*a["pointA"]) for a in gt_annotations]
    pred_post = [Vec3D(r.postsyn_x, r.postsyn_y, r.postsyn_z) for _, r in pred_df.iterrows()]
    pred_centroid = [
        Vec3D(float(r.centroid_x), float(r.centroid_y), float(r.centroid_z))
        for _, r in pred_df.iterrows()
    ]

    # --- Look up segment IDs for GT points AND pred centroids ---
    # The synseg centroid is the canonical position+root for the synseg-side
    # anchor (post for synapse_type=postsyn, pre for synapse_type=presyn).
    # The assignment-net's postsyn_xyz/presyn_xyz on that same side can be
    # several voxels off — using it for distance matching can push genuine
    # TPs over the max_dist cutoff.
    seg_lookup_path = watershed_path or segmentation_path
    if seg_lookup_path is None:
        raise ValueError("Either watershed_path or segmentation_path must be provided")
    all_gt_pts = gt_postB + gt_preA
    label = "watershed" if watershed_path else "segmentation"
    print(
        f"Loading {label} for {len(all_gt_pts)} GT points + {len(pred_centroid)} pred centroids..."
    )
    ws_data, ws_start = load_watershed_region(
        seg_lookup_path,
        all_gt_pts + pred_centroid,
        resolution,
    )

    gt_post_sv = [lookup_sv_id(p, ws_data, ws_start) for p in gt_postB]
    gt_pre_sv = [lookup_sv_id(p, ws_data, ws_start) for p in gt_preA]
    pred_centroid_sv = [lookup_sv_id(p, ws_data, ws_start) for p in pred_centroid]

    if cave_client is not None:
        # CAVE pipeline: pred SVs from parquet, resolve all via ChunkedGraph
        pred_post_sv = [int(x) for x in pred_df["postsyn_sv_id"]]
        pred_pre_sv = [int(x) for x in pred_df["presyn_sv_id"]]

        all_svs = gt_post_sv + gt_pre_sv + pred_post_sv + pred_pre_sv + pred_centroid_sv
        print(f"Resolving {len(all_svs)} SVs to root IDs...")
        all_roots = resolve_roots(all_svs, cave_client)

        ng = len(gt_annotations)
        np_ = len(pred_df)
        gt_post_root = all_roots[:ng]
        gt_pre_root = all_roots[ng : 2 * ng]
        pred_post_root = all_roots[2 * ng : 2 * ng + np_]
        pred_pre_root = all_roots[2 * ng + np_ : 2 * ng + 2 * np_]
        pred_centroid_root = all_roots[2 * ng + 2 * np_ :]
    else:
        # Flat segmentation: GT segment IDs from the seg lookup are already
        # final; pred cell IDs come from the parquet presyn_id/postsyn_id
        # columns (cellseg IDs written by the assignment flow).
        gt_post_root = [int(x) for x in gt_post_sv]
        gt_pre_root = [int(x) for x in gt_pre_sv]
        pred_post_root = [int(x) for x in pred_df["postsyn_id"]]
        pred_pre_root = [int(x) for x in pred_df["presyn_id"]]
        pred_centroid_root = [int(sv) for sv in pred_centroid_sv]

    # The synseg-side anchor (centroid + centroid root) overrides the
    # assignment-net's same-side anchor for matching.
    if synapse_type == "presyn":
        pred_pre_root = list(pred_centroid_root)
    else:
        pred_post_root = list(pred_centroid_root)

    # Drop predictions where the assignment net assigned the same cell as
    # both pre and post partner. These are typically synseg blobs that bled
    # across the synaptic cleft, and the assignment net correctly identifies
    # them as same-cell — i.e. not a real synapse to evaluate.
    self_mask = np.array(
        [
            not (pre != 0 and post != 0 and pre == post)
            for pre, post in zip(pred_pre_root, pred_post_root)
        ]
    )
    n_self = int((~self_mask).sum())
    if n_self:
        print(
            f"  Filtered {n_self}/{len(pred_df)} predictions with pre_root == post_root (bled-over synseg)"
        )
        pred_df = pred_df[self_mask].reset_index(drop=True)
        pred_post = [p for p, k in zip(pred_post, self_mask) if k]
        pred_centroid = [p for p, k in zip(pred_centroid, self_mask) if k]
        pred_post_root = [r for r, k in zip(pred_post_root, self_mask) if k]
        pred_pre_root = [r for r, k in zip(pred_pre_root, self_mask) if k]
        pred_centroid_root = [r for r, k in zip(pred_centroid_root, self_mask) if k]

    # Report zero-SV stats
    n_gt = len(gt_annotations)
    n_pred = len(pred_df)
    gt_post_zeros = sum(1 for sv in gt_post_sv if sv == 0)
    gt_pre_zeros = sum(1 for sv in gt_pre_sv if sv == 0)
    if gt_post_zeros:
        print(f"  WARNING: {gt_post_zeros}/{n_gt} GT postsyn points have no segment ID")
    if gt_pre_zeros:
        print(f"  WARNING: {gt_pre_zeros}/{n_gt} GT presyn points have no segment ID")

    # --- Boundary exclusion ---
    # Predictions with centroids in the border margin are excluded from FP counting
    boundary_excluded: set[int] = set()
    if boundary_margin is not None and bbox_st is not None:
        mx, my, mz = boundary_margin
        for j, row in enumerate(pred_df.itertuples()):
            cx, cy, cz = row.centroid_x, row.centroid_y, row.centroid_z
            if not (
                bbox_st[0] + mx <= cx < bbox_end[0] - mx
                and bbox_st[1] + my <= cy < bbox_end[1] - my
                and bbox_st[2] + mz <= cz < bbox_end[2] - mz
            ):
                boundary_excluded.add(j)
        print(
            f"  Boundary margin [{mx}, {my}, {mz}]: excluding {len(boundary_excluded)}/{n_pred} predictions from FP"
        )

    # Convert to nm for distance matching.
    # The synseg-side anchor (post for synapse_type=postsyn, pre for presyn) is
    # the synseg centroid, NOT the assignment-net's postsyn_xyz/presyn_xyz which
    # can be displaced several voxels and falsely push a real TP over the cutoff.
    gt_post_nm = [p * resolution for p in gt_postB]
    gt_pre_nm = [p * resolution for p in gt_preA]
    pred_centroid_nm = [p * resolution for p in pred_centroid]
    if synapse_type == "presyn":
        pred_pre_nm = pred_centroid_nm
        pred_post_nm = [p * resolution for p in pred_post]
    else:
        pred_post_nm = pred_centroid_nm
        pred_pre_nm = [
            Vec3D(float(r.presyn_x), float(r.presyn_y), float(r.presyn_z)) * resolution
            for _, r in pred_df.iterrows()
        ]

    def _apply_boundary_exclusion(stats: dict) -> None:
        """Subtract boundary-excluded unmatched preds from FP and recompute P/R/F1."""
        if not boundary_excluded:
            return
        matched_pred = {pi for _, pi in stats["matches"]}
        excluded_fp = boundary_excluded - matched_pred
        stats["fp"] -= len(excluded_fp)
        stats["fp_pred_indices"] = [
            j for j in stats["fp_pred_indices"] if j not in boundary_excluded
        ]
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        stats["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        stats["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        p, r = stats["precision"], stats["recall"]
        stats["f1"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        if excluded_fp:
            print(f"  After boundary exclusion: {len(excluded_fp)} FP removed")

    # --- Detection evaluation ---
    # Matches and validates the "own" side (the side the synseg blob represents):
    #   synapse_type=postsyn (PST): match post endpoints, validate post_root.
    #   synapse_type=presyn (vesicle/ribbon): match pre endpoints, validate pre_root.
    # Dedup key is the single own-side root — so multiple annotations/predictions
    # of the same event (e.g. multiple ribbon partners) collapse to one detection.
    print("Evaluating detection...")
    if synapse_type == "presyn":
        det_gt_pts = [p * resolution for p in gt_preA]
        det_pred_pts = pred_pre_nm
        det_gt_keys = list(gt_pre_root)
        det_pred_keys = list(pred_pre_root)
    else:
        det_gt_pts = gt_post_nm
        det_pred_pts = pred_post_nm
        det_gt_keys = list(gt_post_root)
        det_pred_keys = list(pred_post_root)

    def detect_valid(gi, pi):
        gr, pr = det_gt_keys[gi], det_pred_keys[pi]
        return gr != 0 and pr != 0 and gr == pr

    detect_stats = match_points(det_gt_pts, det_pred_pts, max_dist_nm, valid_fn=detect_valid)
    _apply_boundary_exclusion(detect_stats)
    _apply_key_dedup(
        detect_stats,
        det_gt_keys,
        det_pred_keys,
        det_gt_pts,
        det_pred_pts,
        max_dist_nm,
        boundary_excluded,
    )

    # --- Assignment evaluation (independent Hungarian pass) ---
    # Validates both pre and post root IDs. Dedup key is the (pre, post) pair.
    print("Evaluating assignment...")

    def assign_valid(gi, pi):
        gr_post, pr_post = gt_post_root[gi], pred_post_root[pi]
        gr_pre, pr_pre = gt_pre_root[gi], pred_pre_root[pi]
        return (
            gr_post != 0
            and pr_post != 0
            and gr_post == pr_post
            and gr_pre != 0
            and pr_pre != 0
            and gr_pre == pr_pre
        )

    assign_stats = match_points(gt_post_nm, pred_post_nm, max_dist_nm, valid_fn=assign_valid)
    _apply_boundary_exclusion(assign_stats)
    assign_gt_keys = [(gt_pre_root[i], gt_post_root[i]) for i in range(len(gt_pre_root))]
    assign_pred_keys = [(pred_pre_root[i], pred_post_root[i]) for i in range(len(pred_pre_root))]
    _apply_key_dedup(
        assign_stats,
        assign_gt_keys,
        assign_pred_keys,
        gt_post_nm,
        pred_post_nm,
        max_dist_nm,
        boundary_excluded,
    )

    context = {
        "gt_pre_root": gt_pre_root,
        "gt_post_root": gt_post_root,
        "pred_pre_root": pred_pre_root,
        "pred_post_root": pred_post_root,
        "gt_ignored": gt_ignored,
        "gt_annotations": gt_annotations,
        "pred_df": pred_df,
    }

    return detect_stats, assign_stats, context


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_stats(stats: dict, title: str):
    print()
    print(title)
    print("-" * len(title))
    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    print(
        f"  TP={tp}  FP={fp}  FN={fn}  |  P={stats['precision']:.3f}  R={stats['recall']:.3f}  F1={stats['f1']:.3f}"
    )
    n_dups = stats.get("dup_tp", 0) + stats.get("dup_fp", 0) + stats.get("dup_fn", 0)
    if n_dups > 0:
        parts = []
        if stats.get("dup_tp", 0):
            parts.append(f"dup_pred_tp={stats['dup_tp']}")
        if stats.get("dup_fp", 0):
            parts.append(f"dup_pred_fp={stats['dup_fp']}")
        dup_fn_m = stats.get("dup_fn_matched", 0)
        dup_fn_u = stats.get("dup_fn_unmatched", 0)
        if dup_fn_m:
            parts.append(f"dup_gt_matched={dup_fn_m}")
        if dup_fn_u:
            parts.append(f"dup_gt_missed={dup_fn_u}")
        if not dup_fn_m and not dup_fn_u and stats.get("dup_fn", 0):
            parts.append(f"dup_gt={stats['dup_fn']}")
        print(f"  Dedup: {', '.join(parts)}")
        print(
            f"  Dedup: P={stats['precision_dedup']:.3f}  R={stats['recall_dedup']:.3f}  F1={stats['f1_dedup']:.3f}"
        )
    if "mean_dist" in stats:
        print(
            f"  Match dist: mean={stats['mean_dist']:.0f}nm  median={stats['median_dist']:.0f}nm  range=[{stats['min_dist']:.0f}-{stats['max_dist']:.0f}]nm"
        )


def build_ng_json(annotations: list[dict], resolution: Vec3D) -> dict:
    return {
        "type": "annotation",
        "source": {
            "url": "local://annotations",
            "transform": {
                "outputDimensions": {
                    "x": [resolution[0] * 1e-9, "m"],
                    "y": [resolution[1] * 1e-9, "m"],
                    "z": [resolution[2] * 1e-9, "m"],
                },
                "inputDimensions": {
                    "0": [resolution[0] * 1e-9, "m"],
                    "1": [resolution[1] * 1e-9, "m"],
                    "2": [resolution[2] * 1e-9, "m"],
                },
            },
        },
        "tool": "annotateLine",
        "annotations": annotations,
    }


def write_outputs(
    output_dir: str,
    enriched: dict[str, list[dict]],
    resolution: Vec3D,
):
    os.makedirs(output_dir, exist_ok=True)
    for label, anns in enriched.items():
        path = os.path.join(output_dir, f"{label}_lines.json")
        with open(path, "w") as f:
            json.dump(build_ng_json(anns, resolution), f, indent=2)
        print(f"  Wrote {len(anns)} annotations to {path}")


def _assign_score_field(synapse_type: str) -> str:
    """Column whose value is the network's assignment confidence for this mode.

    presyn (vesicle/ribbon) → post side is from the network
    postsyn (PST)            → pre side is from the network
    cleft                    → both, fall back to post (the more informative side
                               in most current setups)
    """
    return "presyn_assign_score" if synapse_type == "postsyn" else "postsyn_assign_score"


def _row_scores(row, synapse_type: str) -> tuple[float | None, float | None]:
    """Return (pred_score, assign_score) for a parquet row, or None if absent/NaN."""
    pred_score = None
    if hasattr(row, "mean_score") and not pd.isna(row.mean_score):
        pred_score = float(row.mean_score)
    assign_score = None
    field = _assign_score_field(synapse_type)
    if hasattr(row, field):
        v = getattr(row, field)
        if not pd.isna(v):
            assign_score = float(v)
    return pred_score, assign_score


def _enrich_pred_annotation(
    row,
    ann_id: str,
    pre_root: int,
    post_root: int,
    synapse_type: str = "postsyn",
    sort_by: str = "pred_score",
) -> dict:
    desc_parts = [f"SynID: {int(row.syn_id)}"]
    pred_score, assign_score = _row_scores(row, synapse_type)
    if pred_score is not None:
        desc_parts.append(f"Pred: {pred_score:.4f}")
    if assign_score is not None:
        desc_parts.append(f"Asn: {assign_score:.4f}")
    score = pred_score
    if sort_by == "assign_score":
        score = assign_score
    # For PST (postsyn) predictions: centroid is on the post cell, presyn_xyz is
    # the anchor on the pre cell → line goes presyn(A) → centroid=post(B).
    # For vesicle cloud (presyn) predictions: centroid is on the pre cell,
    # postsyn_xyz is the anchor on the post cell → line goes centroid=pre(A) →
    # postsyn(B).
    if synapse_type == "presyn":
        pointA = [float(row.centroid_x), float(row.centroid_y), float(row.centroid_z) + 0.5]
        pointB = [float(row.postsyn_x), float(row.postsyn_y), float(row.postsyn_z) + 0.5]
    else:
        pointA = [float(row.presyn_x), float(row.presyn_y), float(row.presyn_z) + 0.5]
        pointB = [float(row.centroid_x), float(row.centroid_y), float(row.centroid_z) + 0.5]
    ann = {
        "pointA": pointA,
        "pointB": pointB,
        "type": "line",
        "id": ann_id,
        "description": " | ".join(desc_parts),
    }
    if pre_root != 0 or post_root != 0:
        ann["segments"] = [[str(pre_root), str(post_root)]]
    ann["_sort_key"] = score if score is not None else 0.0
    return ann


def _enrich_gt_annotation(
    gt_ann: dict,
    pre_root: int,
    post_root: int,
    matched_pred_row=None,
    synapse_type: str = "postsyn",
    sort_by: str = "pred_score",
) -> dict:
    # Keep only the fields Neuroglancer expects for line annotations. Copying
    # everything from GT can leak stray "props" / "relationships" / unknown keys
    # that NG silently drops annotations over.
    ann = {
        "pointA": list(gt_ann["pointA"]),
        "pointB": list(gt_ann["pointB"]),
        "type": gt_ann.get("type", "line"),
        "id": gt_ann.get("id"),
    }
    if "description" in gt_ann:
        ann["description"] = gt_ann["description"]
    desc_parts = []
    score = None
    if matched_pred_row is not None:
        desc_parts.append(f"SynID: {int(matched_pred_row.syn_id)}")
        pred_score, assign_score = _row_scores(matched_pred_row, synapse_type)
        if pred_score is not None:
            desc_parts.append(f"Pred: {pred_score:.4f}")
        if assign_score is not None:
            desc_parts.append(f"Asn: {assign_score:.4f}")
        score = assign_score if sort_by == "assign_score" else pred_score
    if desc_parts:
        ann["description"] = " | ".join(desc_parts)
    if pre_root != 0 or post_root != 0:
        ann["segments"] = [[str(pre_root), str(post_root)]]
    ann["_sort_key"] = score if score is not None else 0.0
    return ann


def build_enriched_annotations(
    assign_stats: dict,
    context: dict,
    synapse_type: str = "postsyn",
    sort_by: str = "pred_score",
) -> dict[str, list[dict]]:
    """Build enriched annotation lists with description, segments, sorted by score.

    sort_by: "pred_score" sorts by mean_score (synseg-net confidence over the cleft);
             "assign_score" sorts by the assignment-net confidence on the partner cell.
    """
    gt_annotations = context["gt_annotations"]
    pred_df = context["pred_df"]
    gt_pre_root = context["gt_pre_root"]
    gt_post_root = context["gt_post_root"]
    pred_pre_root = context["pred_pre_root"]
    pred_post_root = context["pred_post_root"]

    # TP: GT annotations that matched
    tp = []
    for gi, pi in assign_stats["matches"]:
        tp.append(
            _enrich_gt_annotation(
                gt_annotations[gi],
                gt_pre_root[gi],
                gt_post_root[gi],
                matched_pred_row=pred_df.iloc[pi],
                synapse_type=synapse_type,
                sort_by=sort_by,
            )
        )

    # FP: unmatched predictions
    fp = []
    for j in assign_stats["fp_pred_indices"]:
        fp.append(
            _enrich_pred_annotation(
                pred_df.iloc[j],
                f"fp_{j}",
                pred_pre_root[j],
                pred_post_root[j],
                synapse_type=synapse_type,
                sort_by=sort_by,
            )
        )

    # FN: unmatched GT
    fn = []
    for i in assign_stats["fn_gt_indices"]:
        fn.append(
            _enrich_gt_annotation(
                gt_annotations[i],
                gt_pre_root[i],
                gt_post_root[i],
                synapse_type=synapse_type,
                sort_by=sort_by,
            )
        )

    # Dup TP
    dup_tp = []
    for j in assign_stats.get("dup_tp_pred_indices", []):
        dup_tp.append(
            _enrich_pred_annotation(
                pred_df.iloc[j],
                f"dup_tp_{j}",
                pred_pre_root[j],
                pred_post_root[j],
                synapse_type=synapse_type,
                sort_by=sort_by,
            )
        )

    # Dup FP (redundant FP predictions with same pair nearby)
    dup_fp = []
    for j in assign_stats.get("dup_fp_pred_indices", []):
        dup_fp.append(
            _enrich_pred_annotation(
                pred_df.iloc[j],
                f"dup_fp_{j}",
                pred_pre_root[j],
                pred_post_root[j],
                synapse_type=synapse_type,
                sort_by=sort_by,
            )
        )

    # Dup FN matched (GT dup where the synapse WAS correctly found)
    dup_fn_matched = []
    for i in assign_stats.get(
        "dup_fn_matched_gt_indices", assign_stats.get("dup_fn_gt_indices", [])
    ):
        dup_fn_matched.append(
            _enrich_gt_annotation(
                gt_annotations[i],
                gt_pre_root[i],
                gt_post_root[i],
                synapse_type=synapse_type,
                sort_by=sort_by,
            )
        )

    # Dup FN unmatched (GT dup where the synapse was missed, but only count once)
    dup_fn_unmatched = []
    for i in assign_stats.get("dup_fn_unmatched_gt_indices", []):
        dup_fn_unmatched.append(
            _enrich_gt_annotation(
                gt_annotations[i],
                gt_pre_root[i],
                gt_post_root[i],
                synapse_type=synapse_type,
                sort_by=sort_by,
            )
        )

    # Ignored GT (outside bbox — not evaluated, shown for reference)
    gt_ignored_anns = list(context.get("gt_ignored", []))

    # Sort each by score descending, then strip the sort key
    all_lists = [tp, fp, fn, dup_tp, dup_fp, dup_fn_matched, dup_fn_unmatched]
    for lst in all_lists:
        lst.sort(key=lambda a: a.get("_sort_key", 0.0), reverse=True)
        for a in lst:
            a.pop("_sort_key", None)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "dup_tp": dup_tp,
        "dup_fp": dup_fp,
        "dup_fn_matched": dup_fn_matched,
        "dup_fn_unmatched": dup_fn_unmatched,
        "gt_ignored": gt_ignored_anns,
    }


LINE_SHADER = (
    "void main() {\n"
    "  setColor(defaultColor());\n"
    "  setEndpointMarkerSize(2.0, 6.0);\n"
    "  setLineWidth(2.0);\n"
    "}\n"
)


def _local_annotation_layer(
    name: str,
    annotations: list[dict],
    resolution: Vec3D,
    color: str | None = None,
    visible: bool = True,
) -> dict:
    layer: dict = {
        "type": "annotation",
        "source": {
            "url": "local://annotations",
            "transform": {
                "outputDimensions": {
                    "x": [resolution[0] * 1e-9, "m"],
                    "y": [resolution[1] * 1e-9, "m"],
                    "z": [resolution[2] * 1e-9, "m"],
                },
            },
        },
        "tab": "annotations",
        "linkedSegmentationLayer": {"segments": "neuron segmentation"},
        "annotations": annotations,
        "shader": LINE_SHADER,
        "name": name,
    }
    if color:
        layer["annotationColor"] = color
    if not visible:
        layer["visible"] = False
    return layer


def build_ng_state(
    resolution: Vec3D,
    position: list[float],
    image_path: str,
    pcg_source: str,
    seg_path: str,
    lines_path: str,
    enriched: dict[str, list[dict]],
) -> dict:
    tp_anns = enriched["tp"]
    fp_anns = enriched["fp"]
    fn_anns = enriched["fn"]
    dup_tp_anns = enriched["dup_tp"]
    dup_fp_anns = enriched.get("dup_fp", [])
    dup_fn_matched_anns = enriched.get("dup_fn_matched", [])
    dup_fn_unmatched_anns = enriched.get("dup_fn_unmatched", [])
    gt_ignored_anns = enriched.get("gt_ignored", [])

    dims = {
        "x": [resolution[0] * 1e-9, "m"],
        "y": [resolution[1] * 1e-9, "m"],
        "z": [resolution[2] * 1e-9, "m"],
    }

    layers = [
        {"type": "image", "source": image_path + "/|neuroglancer-precomputed:", "name": "EM"},
        {
            "type": "segmentation",
            "source": pcg_source,
            "selectedAlpha": 0.15,
            "segments": [],
            "name": "neuron segmentation",
            "visible": False,
        },
        {
            "type": "segmentation",
            "source": seg_path + "/|neuroglancer-precomputed:",
            "segments": [],
            "name": "Pred Seg",
        },
        {
            "type": "annotation",
            "source": lines_path + "/|neuroglancer-precomputed:",
            "tab": "annotations",
            "shader": LINE_SHADER,
            "name": "Predicted Lines",
            "visible": False,
            "archived": True,
        },
    ]

    # GT (ignored) comes before TP
    if gt_ignored_anns:
        layers.append(
            _local_annotation_layer(
                "GT (ignored)", gt_ignored_anns, resolution, color="#888888", visible=False
            )
        )

    layers.append(_local_annotation_layer("TP", tp_anns, resolution))

    # FP with tagging tool bindings
    fp_layer = _local_annotation_layer("FP", fp_anns, resolution, color="#ff0000")
    fp_layer["toolBindings"] = {
        "A": "tagTool_tag0",
        "D": "tagTool_tag1",
        "S": "tagTool_tag2",
    }
    fp_layer["annotationProperties"] = [
        {"id": "tag0", "type": "uint8", "tag": "Actual Synapse"},
        {"id": "tag1", "type": "uint8", "tag": "Not Synapse"},
        {"id": "tag2", "type": "uint8", "tag": "Can't tell"},
    ]
    # When a layer declares annotationProperties, every annotation must carry a
    # matching `props` array or NG silently drops it.
    for a in fp_layer["annotations"]:
        a.setdefault("props", [0, 0, 0])
    layers.append(fp_layer)

    layers.append(_local_annotation_layer("FN", fn_anns, resolution, color="#0000ff"))

    if dup_tp_anns:
        layers.append(
            _local_annotation_layer("Dup Pred (TP)", dup_tp_anns, resolution, color="#ff8800")
        )
    if dup_fp_anns:
        layers.append(
            _local_annotation_layer("Dup Pred (FP)", dup_fp_anns, resolution, color="#cc4444")
        )
    if dup_fn_matched_anns:
        layers.append(
            _local_annotation_layer(
                "Dup GT (matched)", dup_fn_matched_anns, resolution, color="#ffffaa"
            )
        )
    if dup_fn_unmatched_anns:
        layers.append(
            _local_annotation_layer(
                "Dup GT (missed)", dup_fn_unmatched_anns, resolution, color="#8888ff"
            )
        )

    return {
        "dimensions": dims,
        "position": position,
        "crossSectionScale": 0.18,
        "projectionScale": 220,
        "layers": layers,
        "showSlices": False,
        "layout": "4panel-alt",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate synapse line predictions against GT")
    parser.add_argument("--gt-json", required=True, help="Path to GT annotation JSON")
    parser.add_argument(
        "--pred-metadata", required=True, help="GCS path to predicted metadata parquets"
    )
    parser.add_argument(
        "--watershed-path", default=None, help="GCS path to watershed layer (CAVE pipeline)"
    )
    parser.add_argument(
        "--segmentation-path",
        default=None,
        help="GCS path to flat segmentation layer (no CAVE/ChunkedGraph). "
        "Mutually exclusive with --watershed-path.",
    )
    parser.add_argument(
        "--cave-datastack",
        default=None,
        help="CAVE datastack name (required when using --watershed-path)",
    )
    parser.add_argument(
        "--cave-server",
        default="https://proofreading.zetta.ai",
        help="CAVE server address",
    )
    parser.add_argument(
        "--cave-token",
        default=None,
        help="CAVE auth token (or set CAVE_TOKEN env var). Falls back to the "
        "default secret file when neither is provided.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        nargs=3,
        default=[16, 16, 42],
        help="Processing resolution in nm (default: 16 16 42)",
    )
    parser.add_argument(
        "--max-distance-nm",
        type=float,
        default=600,
        help="Max matching distance in nm (default: 300)",
    )
    parser.add_argument(
        "--boundary-margin",
        type=int,
        nargs=3,
        default=None,
        metavar=("MX", "MY", "MZ"),
        help="Exclude predictions with centroid within this many voxels of bbox edge from FP",
    )
    parser.add_argument("--output-dir", default=".", help="Directory for output JSONs")
    parser.add_argument(
        "--synapse-type",
        choices=["postsyn", "presyn"],
        default="postsyn",
        help="Which side the synseg blob represents. 'postsyn' = PST (centroid on post cell, "
        "line drawn presyn_xyz→centroid). 'presyn' = vesicle cloud (centroid on pre cell, "
        "line drawn centroid→postsyn_xyz).",
    )
    parser.add_argument(
        "--sort-by",
        choices=["pred_score", "assign_score"],
        default="pred_score",
        help="Sort key for predictions in the NG state. 'pred_score' = synseg-net mean over "
        "the cleft (mean_score column). 'assign_score' = assignment-net mean over the "
        "network-driven partner cell (postsyn_assign_score for synapse-type=presyn, "
        "presyn_assign_score for synapse-type=postsyn).",
    )
    parser.add_argument(
        "--upload-state",
        action="store_true",
        help="Build and upload a Neuroglancer state with all layers + TP/FP/FN",
    )
    parser.add_argument(
        "--image-path",
        default=None,
        help="GCS path for EM image layer (no default; pass it explicitly per dataset)",
    )
    parser.add_argument(
        "--pcg-source",
        default=None,
        help="Graphene source URL for PCG segmentation. Defaults to "
        "--watershed-path / --segmentation-path when not provided.",
    )
    return parser, parser.parse_args()


def main():
    parser, args = parse_args()
    resolution = Vec3D(*args.resolution)

    print("Loading ground truth...")
    gt_annotations, gt_res = load_gt(args.gt_json)
    print(f"  {len(gt_annotations)} GT lines at resolution {gt_res}")
    if gt_res != resolution:
        print(f"  WARNING: GT resolution {gt_res} != processing resolution {resolution}")

    print("Loading predictions...")
    pred_df = load_predictions(args.pred_metadata)
    print(f"  {len(pred_df)} predicted assignments")

    if not args.watershed_path and not args.segmentation_path:
        parser.error("Either --watershed-path or --segmentation-path is required")

    cave_client = None
    if args.watershed_path:
        print("Connecting to CAVE...")
        if not args.cave_datastack:
            parser.error("--cave-datastack is required when using --watershed-path")
        token = args.cave_token or os.environ.get("CAVE_TOKEN")
        cave_kwargs = {
            "datastack_name": args.cave_datastack,
            "server_address": args.cave_server,
        }
        if token:
            cave_kwargs["auth_token"] = token
        cave_client = CAVEclient(**cave_kwargs)
    else:
        print("Using flat segmentation (no CAVE/ChunkedGraph)...")

    detect_stats, assign_stats, context = evaluate(
        gt_annotations,
        pred_df,
        args.watershed_path,
        cave_client,
        resolution,
        args.max_distance_nm,
        boundary_margin=tuple(args.boundary_margin) if args.boundary_margin else None,
        metadata_path=args.pred_metadata,
        segmentation_path=args.segmentation_path,
        synapse_type=args.synapse_type,
    )

    print_stats(detect_stats, "DETECTION ACCURACY")
    print_stats(assign_stats, "ASSIGNMENT ACCURACY")

    enriched = build_enriched_annotations(
        assign_stats, context, synapse_type=args.synapse_type, sort_by=args.sort_by
    )

    print("\nWriting output annotations...")
    write_outputs(args.output_dir, enriched, resolution)

    if args.upload_state:
        # Derive seg/lines paths from metadata path
        assign_base = args.pred_metadata.rsplit("/metadata", 1)[0]
        seg_path = assign_base + "/seg"
        lines_path = assign_base + "/lines"

        # Compute center position from prediction centroids
        cx = float(pred_df["centroid_x"].mean())
        cy = float(pred_df["centroid_y"].mean())
        cz = float(pred_df["centroid_z"].mean()) + 0.5
        position = [cx, cy, cz]

        # Fall back to whatever cellseg the eval was actually run against so
        # the NG state shows the right neuron segmentation by default.
        pcg_source = args.pcg_source or args.watershed_path or args.segmentation_path

        print("\nBuilding Neuroglancer state...")
        ng_state = build_ng_state(
            resolution=resolution,
            position=position,
            image_path=args.image_path,
            pcg_source=pcg_source,
            seg_path=seg_path,
            lines_path=lines_path,
            enriched=enriched,
        )

        # Save locally
        state_path = os.path.join(args.output_dir, "ng_state.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(state_path, "w") as f:
            json.dump(ng_state, f, indent=2)
        print(f"  Saved state to {state_path}")

        # Upload to CAVE state server (works even without a datastack)
        upload_client = cave_client
        if upload_client is None:
            token = args.cave_token or os.environ.get("CAVE_TOKEN")
            cave_kwargs = {"server_address": args.cave_server}
            if token:
                cave_kwargs["auth_token"] = token
            upload_client = CAVEclient(**cave_kwargs)
        state_id = upload_client.state.upload_state_json(ng_state)
        server = args.cave_server
        print(
            f"  NG link: https://spelunker.cave-explorer.org/#!middleauth+{server}/nglstate/api/v1/{state_id}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
