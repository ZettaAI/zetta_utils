"""
Ribbon ASSIGNMENT baseline evaluator (no GPU, no assignment net).

For each predicted ribbon CC in the synseg that has at least one GT line
annotation landing inside it (within --pointA-tolerance-nm), determine the
GT partner cell set (dedup'd by post-cell ID) and compare to the dumb
nearest-neighbour baseline (dilate the ribbon mask in nm-space and pick
the closest --max-neighbors cells within --max-radius-nm). Predicted
ribbons with no GT line on them are excluded — they're either false
detections or unlabelled real ribbons; either way they have no reliable
assignment ground truth.

Outputs aggregate P/R/F1 plus TP/FP/FN line annotations + an NG state
showing each match (TP/FN reuse the original GT line; FP is drawn from
the ribbon centroid to the closest voxel of the wrongly-predicted cell).

Usage:
    python scripts/synapse_evaluation/eval_multi_partner.py \\
        --gt-json specs/.../gt009_test.json \\
        --synseg-path gs://.../sweep/thr_0.0800/synseg \\
        --cellseg-path gs://stroeh_sem_mouse_retina_scratch/.../dacey_v1.1_... \\
        --bbox-start 43000 35330 465 \\
        --bbox-end   43315 35610 2035 \\
        --resolution 16 16 40 \\
        --max-radius-nm 350 \\
        --max-neighbors 3 \\
        --pointA-tolerance-nm 50 \\
        --upload-state
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

from eval_synapses import (
    LINE_SHADER,
    _local_annotation_layer,
    load_gt,
    load_predictions,
    resolve_roots,
)

from zetta_utils.geometry import Vec3D
from zetta_utils.internal.synapses.syn_assignment import _select_nearest_via_dilation
from zetta_utils.layer.volumetric.cloudvol import build_cv_layer


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _read_window(layer, start_vx, end_vx, resolution: Sequence[float]) -> np.ndarray:
    res = Vec3D(*resolution)
    sx, sy, sz = start_vx
    ex, ey, ez = end_vx
    return layer[res, slice(sx, ex), slice(sy, ey), slice(sz, ez)][0]


def _vox_floor(p: Sequence[float], origin: Sequence[float]) -> tuple[int, int, int]:
    return (
        int(math.floor(p[0] - origin[0])),
        int(math.floor(p[1] - origin[1])),
        int(math.floor(p[2] - origin[2])),
    )


def _ribbon_at_pointA(
    pointA: Sequence[float],
    synseg: np.ndarray,
    origin: Sequence[float],
    voxel_res: Sequence[float],
    tolerance_nm: float,
) -> int:
    """Return the synseg ID at pointA (or the closest non-zero ID within
    tolerance_nm). 0 = no ribbon found.
    """
    li = _vox_floor(pointA, origin)
    H, W, D = synseg.shape
    if not (0 <= li[0] < H and 0 <= li[1] < W and 0 <= li[2] < D):
        return 0
    val = int(synseg[li])
    if val != 0:
        return val
    rx = int(math.ceil(tolerance_nm / voxel_res[0]))
    ry = int(math.ceil(tolerance_nm / voxel_res[1]))
    rz = int(math.ceil(tolerance_nm / voxel_res[2]))
    x0, x1 = max(0, li[0] - rx), min(H, li[0] + rx + 1)
    y0, y1 = max(0, li[1] - ry), min(W, li[1] + ry + 1)
    z0, z1 = max(0, li[2] - rz), min(D, li[2] + rz + 1)
    sub = synseg[x0:x1, y0:y1, z0:z1]
    nz = np.argwhere(sub > 0)
    if len(nz) == 0:
        return 0
    target_local = np.array([li[0] - x0, li[1] - y0, li[2] - z0])
    diffs_nm = (nz - target_local) * np.array(voxel_res)
    dists_nm = np.sqrt((diffs_nm ** 2).sum(axis=1))
    in_range = dists_nm <= tolerance_nm
    if not in_range.any():
        return 0
    best = nz[in_range][np.argmin(dists_nm[in_range])]
    return int(sub[best[0], best[1], best[2]])


# ---------------------------------------------------------------------------
# Annotation builders
# ---------------------------------------------------------------------------


def _z_align(z: float) -> float:
    """NG quirk: annotations look right when z = floor(z) + 0.5."""
    return math.floor(float(z)) + 0.5


def _line_ann(
    pointA: Sequence[float],
    pointB: Sequence[float],
    ann_id: str,
    description: str,
    pre_id: int = 0,
    post_id: int = 0,
) -> dict:
    ann = {
        "type": "line",
        "id": ann_id,
        "pointA": [float(pointA[0]), float(pointA[1]), _z_align(pointA[2])],
        "pointB": [float(pointB[0]), float(pointB[1]), _z_align(pointB[2])],
        "description": description,
    }
    if pre_id != 0 or post_id != 0:
        ann["segments"] = [[str(pre_id), str(post_id)]]
    return ann


def _ribbon_pre_anchor(syn_mask_local: np.ndarray, local_origin: Sequence[int]) -> Vec3D:
    """Pick a representative voxel of the ribbon for use as the line's pointA:
    closest in-mask voxel to the geometric centroid.
    """
    coords = np.argwhere(syn_mask_local)
    com = coords.mean(axis=0)
    best = coords[np.argmin(np.sum((coords - com) ** 2, axis=1))]
    return Vec3D(*(int(best[d] + local_origin[d]) for d in range(3)))


def _closest_cell_voxel(
    cellseg_local: np.ndarray,
    cell_id: int,
    syn_mask_local: np.ndarray,
    local_origin: Sequence[int],
    voxel_res: Sequence[float],
) -> tuple["Vec3D | None", float]:
    """Closest voxel of `cell_id` (in physical nm) to any syn-mask voxel.
    Returns (voxel, distance_nm); voxel is None if the cell isn't in the
    local crop (and distance_nm is then float('inf')).
    """
    cell_mask = cellseg_local == cell_id
    if not cell_mask.any():
        return None, float("inf")
    syn_coords = np.argwhere(syn_mask_local)
    cell_coords = np.argwhere(cell_mask)
    res_arr = np.array(voxel_res)
    tree = cKDTree(syn_coords * res_arr)
    dists, _ = tree.query(cell_coords * res_arr, k=1)
    best_idx = int(np.argmin(dists))
    best = cell_coords[best_idx]
    voxel = Vec3D(*(int(best[d] + local_origin[d]) for d in range(3)))
    return voxel, float(dists[best_idx])


# ---------------------------------------------------------------------------
# NG state
# ---------------------------------------------------------------------------


def build_ng_state(
    resolution: Vec3D,
    position: list[float],
    image_path: str,
    pcg_source: str,
    ribbon_seg_path: str,
    tp_anns: list[dict],
    fp_anns: list[dict],
    fn_anns: list[dict],
) -> dict:
    dims = {
        "x": [resolution[0] * 1e-9, "m"],
        "y": [resolution[1] * 1e-9, "m"],
        "z": [resolution[2] * 1e-9, "m"],
    }
    layers: list[dict] = [
        {"type": "image", "source": image_path + "/|neuroglancer-precomputed:", "name": "EM"},
        {
            "type": "segmentation",
            "source": pcg_source,
            "selectedAlpha": 0.15,
            "segments": [],
            "name": "neuron segmentation",
        },
        {
            "type": "segmentation",
            "source": ribbon_seg_path + "/|neuroglancer-precomputed:",
            "segments": [],
            "name": "ribbon segmentation",
        },
    ]
    layers.append(_local_annotation_layer("TP", tp_anns, resolution, color="#ffff00"))
    layers.append(_local_annotation_layer("FP", fp_anns, resolution, color="#ff0000"))
    layers.append(_local_annotation_layer("FN", fn_anns, resolution, color="#0000ff"))
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
    p = argparse.ArgumentParser(description="Ribbon assignment baseline evaluation.")
    p.add_argument("--gt-json", required=True)
    p.add_argument(
        "--synseg-path", required=True, help="Predicted ribbon segmentation (synseg layer)."
    )
    p.add_argument(
        "--cellseg-path",
        required=True,
        help="Cell segmentation (used for pre/post cell ID lookup and "
        "the dilate-nearest baseline).",
    )
    p.add_argument(
        "--watershed-path",
        default=None,
        help="Optional watershed/SV layer. When given (with --cave-datastack), "
        "annotation `segments` fields are populated with graphene root IDs "
        "(SV at line endpoint → root via CAVE) so they match the NG state's "
        "linked PCG segmentation. Without this, segments use flat cellseg IDs, "
        "which won't resolve correctly when the linked layer is graphene.",
    )
    p.add_argument(
        "--match-by-root",
        action="store_true",
        help="Match GT vs predicted partners by graphene root ID instead of "
        "flat cellseg ID. Recommended when the cellseg is finer than the "
        "PCG (e.g. dacey_v1.1) — otherwise multiple cellseg fragments under "
        "the same root count as separate partners. Requires --watershed-path.",
    )
    p.add_argument("--bbox-start", type=int, nargs=3, required=True)
    p.add_argument("--bbox-end", type=int, nargs=3, required=True)
    p.add_argument("--resolution", type=float, nargs=3, default=[16, 16, 40])
    p.add_argument(
        "--max-radius-nm",
        type=float,
        default=350.0,
        help="Baseline mode: cap on partner-search radius (nm). "
        "Ignored when --pred-metadata is given.",
    )
    p.add_argument(
        "--max-neighbors",
        type=int,
        default=3,
        help="Baseline mode: max number of partner cells per ribbon. "
        "Ignored when --pred-metadata is given.",
    )
    p.add_argument(
        "--pred-metadata",
        default=None,
        help="Parquet path produced by the assignment flow. When set, "
        "(1) only ribbon CCs present in the parquet are evaluated "
        "(strict apples-to-apples filter — drops ribbons removed by "
        "the assignment flow's merge+size_thr) and (2) partners come "
        "from postsyn_id in the parquet, unless --baseline-partners.",
    )
    p.add_argument(
        "--baseline-partners",
        action="store_true",
        help="Use the dilate-nearest baseline for partner selection even "
        "when --pred-metadata is given (so the parquet acts as a "
        "ribbon-eligibility filter only). Lets you compare baseline "
        "vs network on identical ribbon sets.",
    )
    p.add_argument(
        "--pointA-tolerance-nm",
        type=float,
        default=50.0,
        help="If GT pointA isn't directly inside a synseg CC, search "
        "this far (nm) for the nearest non-zero synseg voxel.",
    )
    p.add_argument(
        "--minimum-size",
        type=int,
        default=0,
        help="Drop synseg CCs smaller than this many voxels before "
        "evaluation (matches the segmentation's --minimum-size).",
    )
    p.add_argument("--output-dir", default=".")
    p.add_argument("--upload-state", action="store_true")
    p.add_argument("--image-path", default="gs://stroeh_sem_mouse_retina/image/v2")
    p.add_argument(
        "--pcg-source",
        default="graphene://middleauth+https://minnie.microns-daf.com/segmentation/table/stroeh_mouse_retina",
        help="Graphene-backed cell-segmentation source for the NG state.",
    )
    p.add_argument(
        "--ribbon-seg-path",
        default=None,
        help="Precomputed ribbon-segmentation layer path to display in the NG "
        "state. Defaults to the sibling 'assignment/seg' next to "
        "--synseg-path (i.e. the merged seg from the assignment flow).",
    )
    p.add_argument("--cave-server", default="https://global.daf-apis.com")
    p.add_argument(
        "--cave-datastack",
        default="stroeh_mouse_retina",
        help="CAVE datastack used to resolve watershed SVs to PCG roots.",
    )
    p.add_argument("--cave-token", default=None)
    return p.parse_args()


def _resolve_segments_to_roots(
    annotations: list[dict],
    watershed: np.ndarray,
    origin: Sequence[int],
    cave_client,
) -> None:
    """In-place: rewrite each annotation's `segments` field to graphene root
    IDs by looking up the watershed SV at pointA (pre side) and pointB (post
    side) and resolving via CAVE. Annotations without watershed coverage
    (out-of-bbox, or SV=0) get segments=[[0, 0]]; harmless for NG.
    """
    if not annotations:
        return
    H, W, D = watershed.shape

    def _sv_at(pt: Sequence[float]) -> int:
        li = _vox_floor(pt, origin)
        if 0 <= li[0] < H and 0 <= li[1] < W and 0 <= li[2] < D:
            return int(watershed[li])
        return 0

    pre_svs: list[int] = []
    post_svs: list[int] = []
    for a in annotations:
        pre_svs.append(_sv_at(a["pointA"]))
        post_svs.append(_sv_at(a["pointB"]))

    all_svs = pre_svs + post_svs
    roots = resolve_roots(all_svs, cave_client)
    n = len(annotations)
    pre_roots = roots[:n]
    post_roots = roots[n:]
    for a, pr, po in zip(annotations, pre_roots, post_roots):
        a["segments"] = [[str(pr), str(po)]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    resolution: Vec3D = Vec3D(*args.resolution)
    voxel_res = tuple(args.resolution)

    print("Loading GT...")
    gt_annotations, gt_res = load_gt(args.gt_json)
    print(f"  {len(gt_annotations)} GT lines @ {gt_res}")

    bbox_st = list(args.bbox_start)
    bbox_end = list(args.bbox_end)
    print(f"\nReading synseg + cellseg over bbox {bbox_st} -- {bbox_end}")
    synseg = _read_window(build_cv_layer(args.synseg_path), bbox_st, bbox_end, voxel_res).astype(
        np.uint64
    )
    cellseg = _read_window(build_cv_layer(args.cellseg_path), bbox_st, bbox_end, voxel_res).astype(
        np.uint64
    )
    origin = bbox_st

    if args.minimum_size > 0:
        ids, counts = np.unique(synseg, return_counts=True)
        small = ids[(ids != 0) & (counts < args.minimum_size)]
        if len(small):
            mask = np.isin(synseg, small)
            synseg = synseg.copy()
            synseg[mask] = 0
            print(f"  Dropped {len(small)} CCs smaller than {args.minimum_size} vx")

    # --- Filter GT to bbox by pointA (the presyn/ribbon side) ---
    gt_in_bbox = []
    for a in gt_annotations:
        pt = a["pointA"]
        if all(bbox_st[d] <= pt[d] < bbox_end[d] for d in range(3)):
            gt_in_bbox.append(a)
    print(f"  GT lines inside bbox (by pointA): {len(gt_in_bbox)}/{len(gt_annotations)}")

    # --- Group GT lines by the ribbon CC their pointA falls on ---
    ribbons_to_gt: dict[int, list[dict]] = defaultdict(list)
    pointA_off_synseg = 0
    pointB_oob = 0
    zero_cell_id = 0

    for a in gt_in_bbox:
        ptA = a["pointA"]
        ptB = a["pointB"]
        syn_id = _ribbon_at_pointA(ptA, synseg, origin, voxel_res, args.pointA_tolerance_nm)
        if syn_id == 0:
            pointA_off_synseg += 1
            continue
        ptA_li = _vox_floor(ptA, origin)
        ptB_li = _vox_floor(ptB, origin)
        H, W, D = cellseg.shape
        if not (0 <= ptA_li[0] < H and 0 <= ptA_li[1] < W and 0 <= ptA_li[2] < D):
            zero_cell_id += 1
            continue
        if not (0 <= ptB_li[0] < H and 0 <= ptB_li[1] < W and 0 <= ptB_li[2] < D):
            pointB_oob += 1
            continue
        pre_id = int(cellseg[ptA_li])
        post_id = int(cellseg[ptB_li])
        if pre_id == 0 or post_id == 0:
            zero_cell_id += 1
            continue
        ribbons_to_gt[syn_id].append(
            {
                "ann": a,
                "pre_id": pre_id,
                "post_id": post_id,
            }
        )

    print(
        f"  GT lines whose pointA missed any ribbon (within {args.pointA_tolerance_nm}nm): "
        f"{pointA_off_synseg}"
    )
    print(f"  GT lines with cellseg==0 at pointA/B: {zero_cell_id}")
    print(f"  GT lines with pointB outside bbox: {pointB_oob}")
    print(f"  Ribbons with at least one GT line: {len(ribbons_to_gt)}")

    # --- If a parquet is given, build syn_id -> {postsyn_id} map. The set of
    # syn_ids in the parquet also acts as a strict eligibility filter so that
    # baseline and network can be compared on identical ribbon sets.
    network_partners: dict[int, set[int]] = {}
    # Per-(syn_id, postsyn_id) mean network-output score (used for "Asn"
    # in the description); 0 when the parquet has no postsyn_assign_score.
    network_partner_score: dict[int, dict[int, float]] = {}
    if args.pred_metadata is not None:
        print(f"\nLoading parquet from {args.pred_metadata} ...")
        pred_df = load_predictions(args.pred_metadata)
        print(f"  {len(pred_df)} parquet rows")
        has_score = "postsyn_assign_score" in pred_df.columns
        for syn_id_val, group in pred_df.groupby("syn_id"):
            # IMPORTANT: never use iterrows() on a frame with uint64 columns —
            # it casts the row to a single dtype Series (float64), and IDs >=
            # 2^53 lose precision (e.g. 77973310461503692 → 77973310461503696).
            # Iterate columns directly via .values to keep native dtypes.
            pid_arr = group["postsyn_id"].values
            posts = {int(pid) for pid in pid_arr if int(pid) != 0}
            network_partners[int(syn_id_val)] = posts
            score_map: dict[int, float] = {}
            if has_score:
                score_arr = group["postsyn_assign_score"].values
                for pid, sc in zip(pid_arr, score_arr):
                    pid_i = int(pid)
                    if pid_i != 0:
                        score_map[pid_i] = float(sc)
            network_partner_score[int(syn_id_val)] = score_map
        if not has_score:
            print(
                "  WARN: parquet has no `postsyn_assign_score` column — "
                "Asn confidences will be omitted."
            )
        before = len(ribbons_to_gt)
        ribbons_to_gt = {
            sid: g for sid, g in ribbons_to_gt.items() if int(sid) in network_partners
        }
        print(f"  Filter to parquet-present ribbons: {len(ribbons_to_gt)}/{before}")

    # --- Per-ribbon: phase 1 (gather pred partners + crops + ANCHORS) ---
    pad_vx = np.ceil(np.array([args.max_radius_nm / r + 5 for r in voxel_res])).astype(int)
    use_network = args.pred_metadata is not None and not args.baseline_partners
    ribbon_data: list[dict] = []
    n_pre_disagree = 0
    for syn_id, entries in ribbons_to_gt.items():
        pre_ids = {e["pre_id"] for e in entries}
        if len(pre_ids) > 1:
            n_pre_disagree += 1
        pre_id = next(iter(pre_ids))
        gt_post_to_ann: dict[int, dict] = {}
        for e in entries:
            gt_post_to_ann.setdefault(e["post_id"], e["ann"])

        full_mask = synseg == syn_id
        coords = np.argwhere(full_mask)
        if len(coords) == 0:
            continue  # filtered out by minimum_size after the GT pointA lookup
        rmin = coords.min(axis=0)
        rmax = coords.max(axis=0) + 1
        cmin = np.maximum(rmin - pad_vx, 0)
        cmax = np.minimum(rmax + pad_vx, np.array(synseg.shape))
        slc = tuple(slice(int(cmin[d]), int(cmax[d])) for d in range(3))
        local_synseg = synseg[slc]
        local_cellseg = cellseg[slc]
        local_syn_mask_bool = local_synseg == syn_id
        local_syn_mask = local_syn_mask_bool.astype(np.float32)

        if use_network:
            pred_post_set = network_partners.get(int(syn_id), set())
        else:
            pred_post_ids = _select_nearest_via_dilation(
                local_cellseg,
                local_syn_mask,
                voxel_res,
                args.max_radius_nm,
                args.max_neighbors,
                exclude_ids={pre_id},
            )
            pred_post_set = set(pred_post_ids)

        # Compute anchors (global voxel coords) once, here. These same anchors
        # are used for (1) the line annotation endpoints, (2) the segments-
        # field watershed lookup, and (3) match-by-root SV resolution. Keeping
        # them shared avoids the inconsistency where different voxels of the
        # same cellseg ID belong to different graphene roots.
        pre_anchor_local = _ribbon_pre_anchor(
            local_syn_mask_bool,
            [int(cmin[d]) for d in range(3)],
        )
        pre_anchor_global = Vec3D(*(int(pre_anchor_local[d] + origin[d]) for d in range(3)))
        # Per-pred anchor: closest cell voxel of that cellseg to the syn mask,
        # and the corresponding nm distance (used as the baseline's "Dist"
        # confidence proxy).
        pred_post_anchors: dict[int, Vec3D | None] = {}
        pred_post_dists: dict[int, float] = {}
        for cid in pred_post_set:
            tgt_local, dist_nm = _closest_cell_voxel(
                local_cellseg,
                cid,
                local_syn_mask_bool,
                [0, 0, 0],
                voxel_res,
            )
            if tgt_local is None:
                pred_post_anchors[int(cid)] = None
            else:
                pred_post_anchors[int(cid)] = Vec3D(
                    *(int(tgt_local[d] + cmin[d] + origin[d]) for d in range(3))
                )
            pred_post_dists[int(cid)] = dist_nm

        ribbon_data.append(
            {
                "syn_id": int(syn_id),
                "pre_id": int(pre_id),
                "gt_post_to_ann": gt_post_to_ann,
                "pred_post_set": pred_post_set,
                "pre_anchor_global": pre_anchor_global,
                "pred_post_anchors": pred_post_anchors,
                "pred_post_dists": pred_post_dists,
            }
        )

    # --- Phase 1.5: optionally resolve roots at the SAME anchor voxels we'll
    # use for line endpoints. This guarantees that the post-key in match-by-
    # root mode equals what the segments field shows in NG. ---
    cellseg_to_root_pre: dict[int, int] = {}
    # Per-ribbon mappings (cellseg → root) using the actual anchor lookups
    rd_gt_post_root: list[dict[int, int]] = []
    rd_pred_post_root: list[dict[int, int]] = []
    if args.match_by_root:
        if not args.watershed_path:
            raise SystemExit("--match-by-root requires --watershed-path (and CAVE access)")
        from caveclient import CAVEclient

        # Load watershed once for the whole bbox.
        ws_full = _read_window(
            build_cv_layer(args.watershed_path), bbox_st, bbox_end, voxel_res
        ).astype(np.uint64)
        H, W, D = ws_full.shape

        def _sv_at_global(pt) -> int:
            li = _vox_floor(pt, origin)
            if 0 <= li[0] < H and 0 <= li[1] < W and 0 <= li[2] < D:
                return int(ws_full[li])
            return 0

        # Collect SVs at all matching-anchor voxels: (a) GT pointA per ribbon
        # (for pre-root display), (b) GT pointB per gt entry, (c) pred anchor
        # per pred cellseg.
        all_svs: set[int] = set()
        for rd in ribbon_data:
            # Pre side: any GT pointA on this ribbon — they should all agree.
            first_ann = next(iter(rd["gt_post_to_ann"].values()))
            sv_pre = _sv_at_global(first_ann["pointA"])
            rd["_pre_sv"] = sv_pre
            if sv_pre:
                all_svs.add(sv_pre)
            # GT post side
            rd["_gt_post_sv"] = {}
            for cid, ann in rd["gt_post_to_ann"].items():
                sv = _sv_at_global(ann["pointB"])
                rd["_gt_post_sv"][cid] = sv
                if sv:
                    all_svs.add(sv)
            # Pred post side
            rd["_pred_post_sv"] = {}
            for cid, anchor in rd["pred_post_anchors"].items():
                sv = _sv_at_global(anchor) if anchor is not None else 0
                rd["_pred_post_sv"][cid] = sv
                if sv:
                    all_svs.add(sv)

        cave_kwargs = {
            "server_address": args.cave_server,
            "datastack_name": args.cave_datastack,
        }
        token = args.cave_token or os.environ.get("CAVE_TOKEN")
        if token:
            cave_kwargs["auth_token"] = token
        cave = CAVEclient(**cave_kwargs)
        sv_list = sorted(all_svs)
        print(f"\nResolving {len(sv_list)} unique SVs at anchor voxels to roots ...")
        if sv_list:
            roots = cave.chunkedgraph.get_roots(sv_list)
            sv_to_root_map = {sv: int(r) for sv, r in zip(sv_list, roots)}
        else:
            sv_to_root_map = {}

        for rd in ribbon_data:
            cellseg_to_root_pre[rd["pre_id"]] = sv_to_root_map.get(rd["_pre_sv"], 0)
            rd_gt_post_root.append(
                {cid: sv_to_root_map.get(sv, 0) for cid, sv in rd["_gt_post_sv"].items()}
            )
            rd_pred_post_root.append(
                {cid: sv_to_root_map.get(sv, 0) for cid, sv in rd["_pred_post_sv"].items()}
            )

    # --- Phase 2: matching + line annotations ---
    total_tp = total_fp = total_fn = 0
    n_ribbons_eval = 0
    tp_anns: list[dict] = []
    fp_anns: list[dict] = []
    fn_anns: list[dict] = []

    for rd_idx, rd in enumerate(ribbon_data):
        syn_id = rd["syn_id"]
        pre_id = rd["pre_id"]
        gt_post_to_ann = rd["gt_post_to_ann"]
        pred_post_set = rd["pred_post_set"]
        pre_anchor_global = rd["pre_anchor_global"]
        pred_post_anchors = rd["pred_post_anchors"]
        pred_post_dists = rd["pred_post_dists"]

        if args.match_by_root:
            # Use roots resolved at the same anchor voxels we use for the
            # line endpoints. Keys (and post-key in description) match what
            # NG's segments field will show.
            gt_post_root = rd_gt_post_root[rd_idx]
            pred_post_root = rd_pred_post_root[rd_idx]
            gt_root_to_cellseg: dict[int, int] = {}
            for cid, r in gt_post_root.items():
                if r != 0:
                    gt_root_to_cellseg.setdefault(r, cid)
            pred_root_to_cellseg: dict[int, int] = {}
            for cid, r in pred_post_root.items():
                if r != 0:
                    pred_root_to_cellseg.setdefault(r, cid)
            gt_keys = set(gt_root_to_cellseg)
            pred_keys = set(pred_root_to_cellseg)
        else:
            gt_keys = set(gt_post_to_ann.keys())
            pred_keys = set(pred_post_set)
            gt_root_to_cellseg = {k: k for k in gt_keys}
            pred_root_to_cellseg = {k: k for k in pred_keys}

        tp_set = gt_keys & pred_keys
        fp_set = pred_keys - gt_keys
        fn_set = gt_keys - pred_keys
        total_tp += len(tp_set)
        total_fp += len(fp_set)
        total_fn += len(fn_set)
        n_ribbons_eval += 1

        # Mean assignment-net confidence over the cellseg post_ids that
        # belong to this matched key (single cellseg in cellseg-mode; one or
        # more cellseg fragments under the same root in match-by-root mode).
        # Returns "" when no parquet / no score column / no entries.
        score_map = network_partner_score.get(syn_id, {}) if use_network else {}

        def _asn_label(key: int, source_keys_to_cellseg: dict[int, int]) -> str:
            if not use_network or not score_map:
                return ""
            if args.match_by_root:
                # All pred cellsegs whose anchor root == key
                cellseg_ids = [c for c, r in pred_post_root.items() if r == key]
            else:
                cellseg_ids = [source_keys_to_cellseg[key]]
            scores = [score_map.get(c, 0.0) for c in cellseg_ids if c in score_map]
            if not scores:
                return ""
            return f" | Asn: {sum(scores) / len(scores):.3f}"

        def _dist_label(key: int, source_keys_to_cellseg: dict[int, int]) -> str:
            """Mean closest-voxel distance (nm) from the ribbon to the matched
            partner's cellseg fragments. Only emitted in baseline mode (since
            the network has its own confidence via Asn)."""
            if use_network:
                return ""
            if args.match_by_root:
                cellseg_ids = [c for c, r in pred_post_root.items() if r == key]
            else:
                cellseg_ids = [source_keys_to_cellseg[key]]
            dists = [
                pred_post_dists[c]
                for c in cellseg_ids
                if c in pred_post_dists and pred_post_dists[c] != float("inf")
            ]
            if not dists:
                return ""
            return f" | Dist: {sum(dists) / len(dists):.0f}nm"

        for key in tp_set:
            cellseg_post = gt_root_to_cellseg[key]
            ann = gt_post_to_ann[cellseg_post]
            tp_anns.append(
                _line_ann(
                    ann["pointA"],
                    ann["pointB"],
                    f"tp_r{syn_id}_p{key}",
                    f"ribbon: {syn_id}"
                    f"{_asn_label(key, pred_root_to_cellseg)}"
                    f"{_dist_label(key, pred_root_to_cellseg)}",
                    pre_id=pre_id,
                    post_id=cellseg_post,
                )
            )
        for key in fn_set:
            cellseg_post = gt_root_to_cellseg[key]
            ann = gt_post_to_ann[cellseg_post]
            fn_anns.append(
                _line_ann(
                    ann["pointA"],
                    ann["pointB"],
                    f"fn_r{syn_id}_p{key}",
                    f"ribbon: {syn_id} (missed)",
                    pre_id=pre_id,
                    post_id=cellseg_post,
                )
            )
        for key in fp_set:
            cellseg_post = pred_root_to_cellseg[key]
            tgt_global = pred_post_anchors.get(cellseg_post)
            if tgt_global is None:
                tgt_global = pre_anchor_global  # safety
            fp_anns.append(
                _line_ann(
                    pre_anchor_global,
                    tgt_global,
                    f"fp_r{syn_id}_p{key}",
                    f"ribbon: {syn_id} (pred only)"
                    f"{_asn_label(key, pred_root_to_cellseg)}"
                    f"{_dist_label(key, pred_root_to_cellseg)}",
                    pre_id=pre_id,
                    post_id=cellseg_post,
                )
            )

    p_ = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    r_ = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) else 0.0

    if args.pred_metadata is not None:
        if args.baseline_partners:
            mode_label = (
                f"baseline (max_radius_nm={args.max_radius_nm}, "
                f"max_neighbors={args.max_neighbors}), filtered to parquet ribbons"
            )
        else:
            mode_label = f"network parquet={args.pred_metadata}"
    else:
        mode_label = (
            f"baseline: max_radius_nm={args.max_radius_nm}, " f"max_neighbors={args.max_neighbors}"
        )
    print()
    print("=" * 70)
    print(f"RIBBON ASSIGNMENT  ({mode_label})")
    print("=" * 70)
    print(f"  Ribbons evaluated:           {n_ribbons_eval}")
    print(f"  Ribbons w/ disagreeing pre:  {n_pre_disagree}")
    print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"  P={p_:.3f}  R={r_:.3f}  F1={f1:.3f}")

    # --- Resolve segments to graphene root IDs (so they match the linked
    # PCG segmentation in the NG state) ---
    if args.watershed_path:
        from caveclient import CAVEclient

        print(f"\nLoading watershed for segment-root resolution ...")
        watershed = _read_window(
            build_cv_layer(args.watershed_path), bbox_st, bbox_end, voxel_res
        ).astype(np.uint64)
        cave_kwargs = {
            "server_address": args.cave_server,
            "datastack_name": args.cave_datastack,
        }
        token = args.cave_token or os.environ.get("CAVE_TOKEN")
        if token:
            cave_kwargs["auth_token"] = token
        cave_client = CAVEclient(**cave_kwargs)
        for anns in (tp_anns, fp_anns, fn_anns):
            _resolve_segments_to_roots(anns, watershed, origin, cave_client)
    else:
        print(
            "\nWARN: --watershed-path not given. Annotation `segments` will hold "
            "flat cellseg IDs that won't resolve in a graphene-PCG-linked NG state."
        )

    # --- Write JSONs ---
    os.makedirs(args.output_dir, exist_ok=True)
    for key, anns in [("tp", tp_anns), ("fp", fp_anns), ("fn", fn_anns)]:
        path = os.path.join(args.output_dir, f"{key}.json")
        with open(path, "w") as f:
            json.dump(anns, f, indent=2)
        print(f"  Wrote {len(anns)} {key} → {path}")

    # --- Optional NG state ---
    if args.upload_state:
        from caveclient import CAVEclient

        cx = (bbox_st[0] + bbox_end[0]) / 2
        cy = (bbox_st[1] + bbox_end[1]) / 2
        cz = (bbox_st[2] + bbox_end[2]) / 2
        position = [cx, cy, cz + 0.5]

        # Default ribbon-seg display layer = the assignment-flow's merged seg
        # sibling to the synseg input.
        if args.ribbon_seg_path is None:
            base = args.synseg_path.rstrip("/").rsplit("/", 1)[0]
            ribbon_seg_path = f"{base}/assignment/seg"
        else:
            ribbon_seg_path = args.ribbon_seg_path

        ng_state = build_ng_state(
            resolution=resolution,
            position=position,
            image_path=args.image_path,
            pcg_source=args.pcg_source,
            ribbon_seg_path=ribbon_seg_path,
            tp_anns=tp_anns,
            fp_anns=fp_anns,
            fn_anns=fn_anns,
        )
        state_path = os.path.join(args.output_dir, "ng_state.json")
        with open(state_path, "w") as f:
            json.dump(ng_state, f, indent=2)
        print(f"  Saved NG state to {state_path}")

        token = args.cave_token or os.environ.get("CAVE_TOKEN")
        cave_kwargs = {"server_address": args.cave_server}
        if token:
            cave_kwargs["auth_token"] = token
        client = CAVEclient(**cave_kwargs)
        state_id = client.state.upload_state_json(ng_state)
        print(
            f"  NG link: https://spelunker.cave-explorer.org/#!middleauth+"
            f"{args.cave_server}/nglstate/api/v1/{state_id}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
