"""
Simple ribbon (or vesicle / PST) DETECTION evaluation.

Compares the per-synapse-segment centroids of a sweep's synseg to a GT
line-annotation JSON. Reports TP / FP / FN counts and writes point
annotations + an (optional) Neuroglancer state with the matches highlighted.
No assignment evaluation, no CAVE root resolution, no synseg-net scoring
required — just "did we find a synapse near each GT line?".

Two ways to source predicted centroids:
  --synseg-path PATH      Load the synseg layer and compute CC centroids
                          directly. This is the *assignment-invariant*
                          detection score; recommended when comparing across
                          different assign_type runs.
  --pred-metadata PATH    Load centroids from the parquet written by the
                          assignment flow. Equivalent to the detection score
                          eval_synapses.py reports, i.e. *after* merge
                          and size_thr.

Usage (ribbons / vesicles, GT line points to the presyn side):
    python scripts/synapse_evaluation/eval_detection.py \\
        --gt-json specs/.../gt009_test.json \\
        --synseg-path gs://.../sweep/thr_0.0800/synseg \\
        --bbox-start 43000 35330 465 \\
        --bbox-end   43315 35610 2035 \\
        --image-path gs://stroeh_sem_mouse_retina/image/v2 \\
        --resolution 16 16 40 \\
        --max-distance-nm 600 \\
        --gt-anchor pointA \\
        --upload-state
"""
from __future__ import annotations

import argparse
import json
import os
import uuid

import fastremap
import numpy as np
import pandas as pd
from scipy import ndimage

# Reuse helpers from eval_synapses.py to avoid duplication.
from eval_synapses import (
    LINE_SHADER,  # only used for the "Pred Lines" reference layer if path is passed
    _apply_key_dedup,
    _local_annotation_layer,
    gsutil_cp,  # noqa: F401  (kept for parity if you extend later)
    load_gt,
    load_predictions,
    load_watershed_region,
    lookup_sv_id,
    match_points,
    print_stats,
)

from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric.cloudvol import build_cv_layer


def predictions_from_synseg(
    synseg_path: str,
    bbox_start: list[int],
    bbox_end: list[int],
    resolution: list[float],
    minimum_size: int = 0,
) -> pd.DataFrame:
    """Compute per-CC centroids and sizes from a synseg layer over the given bbox.

    Returns a DataFrame with columns matching what evaluate_detection() needs:
    syn_id, size, centroid_x/y/z. Optional 'mean_score' is left absent.
    """
    bbox = BBox3D.from_coords(start_coord=bbox_start, end_coord=bbox_end, resolution=resolution)
    layer = build_cv_layer(synseg_path)
    res = Vec3D(*resolution)
    sx, sy, sz = bbox_start
    ex, ey, ez = bbox_end
    print(f"  Reading synseg over bbox {bbox_start}..{bbox_end} at {tuple(res)} nm")
    data = layer[res, slice(sx, ex), slice(sy, ey), slice(sz, ez)][0]  # (X, Y, Z)

    syn_ids, sizes = np.unique(data, return_counts=True)
    nz = syn_ids != 0
    syn_ids = syn_ids[nz]
    sizes = sizes[nz]
    if minimum_size > 0:
        keep = sizes >= minimum_size
        syn_ids = syn_ids[keep]
        sizes = sizes[keep]
    print(f"  {len(syn_ids)} CC components in synseg (min_size={minimum_size})")

    if len(syn_ids) == 0:
        return pd.DataFrame(columns=["syn_id", "size", "centroid_x", "centroid_y", "centroid_z"])

    # ndimage.find_objects allocates a slot per integer up to max_label, so on
    # raw uint64 syn_ids it tries to allocate ~max_id slots and OOMs. Renumber
    # to contiguous small ints first; cast to uint32 to also dodge a known
    # scipy bug that does labels.max()+2 internally and overflows uint8/16.
    compact, remap_dict = fastremap.renumber(data, in_place=False)
    compact = compact.astype(np.uint32)
    orig_to_new = {int(o): int(n) for o, n in remap_dict.items() if n > 0}

    # ndimage.center_of_mass returns the geometric mean of voxel positions —
    # for non-convex shapes (curved ribbons) this can fall *outside* the
    # segment. Snap each centroid to the closest in-segment voxel so points
    # are guaranteed to land inside the synseg.
    new_labels = [orig_to_new[int(sid)] for sid in syn_ids]
    centroids = ndimage.center_of_mass(compact > 0, compact, new_labels)
    bboxes = ndimage.find_objects(compact, max_label=max(new_labels))

    cx_list: list[float] = []
    cy_list: list[float] = []
    cz_list: list[float] = []
    for new_lbl, (lcx, lcy, lcz) in zip(new_labels, centroids):
        slc = bboxes[new_lbl - 1]
        sub = compact[slc]
        coords = np.argwhere(sub == new_lbl)
        if len(coords):
            origin = np.array([slc[0].start, slc[1].start, slc[2].start])
            target = np.array([lcx, lcy, lcz]) - origin
            best = coords[np.argmin(np.sum((coords - target) ** 2, axis=1))]
            lcx, lcy, lcz = (best + origin).tolist()
        cx_list.append(lcx + sx)
        cy_list.append(lcy + sy)
        cz_list.append(lcz + sz)
    return pd.DataFrame(
        {
            "syn_id": syn_ids.astype(np.uint64),
            "size": sizes.astype(np.uint64),
            "centroid_x": np.array(cx_list, dtype=np.float32),
            "centroid_y": np.array(cy_list, dtype=np.float32),
            "centroid_z": np.array(cz_list, dtype=np.float32),
        }
    )


def _gt_point(ann: dict, anchor: str, midpoint: bool) -> Vec3D:
    if midpoint:
        a: Vec3D = Vec3D(*ann["pointA"])
        b: Vec3D = Vec3D(*ann["pointB"])
        return Vec3D(*((a[i] + b[i]) / 2 for i in range(3)))
    return Vec3D(*ann[anchor])


def _filter_to_bbox(pred_df, gt_annotations, bbox_st, bbox_end, anchor: str, midpoint: bool):
    """Return (kept_pred_df, kept_gt, ignored_gt) filtered to bbox [start, end)."""
    if bbox_st is None or bbox_end is None:
        return pred_df, list(gt_annotations), []

    n_before = len(pred_df)
    mask = (
        (pred_df["centroid_x"] >= bbox_st[0])
        & (pred_df["centroid_x"] < bbox_end[0])
        & (pred_df["centroid_y"] >= bbox_st[1])
        & (pred_df["centroid_y"] < bbox_end[1])
        & (pred_df["centroid_z"] >= bbox_st[2])
        & (pred_df["centroid_z"] < bbox_end[2])
    )
    pred_df_kept = pred_df[mask].reset_index(drop=True)
    n_dropped = n_before - len(pred_df_kept)
    if n_dropped:
        print(f"  Dropped {n_dropped} predictions outside bbox")

    gt_kept: list[dict] = []
    gt_ignored: list[dict] = []
    for a in gt_annotations:
        pt = _gt_point(a, anchor, midpoint)
        if all(bbox_st[d] <= pt[d] < bbox_end[d] for d in range(3)):
            gt_kept.append(a)
        else:
            gt_ignored.append(a)
    if gt_ignored:
        label = "midpoint" if midpoint else anchor
        print(
            f"  Filtered GT to bbox ({label}): {len(gt_kept)}/{len(gt_annotations)}"
            f" ({len(gt_ignored)} ignored)"
        )
    return pred_df_kept, gt_kept, gt_ignored


def _make_point_ann(point_xyz, ann_id: str, description: str | None = None) -> dict:
    """Build a Neuroglancer 'point' annotation.

    NG renders point annotations between voxel slices unless z is `.5`-aligned;
    floor(z) + 0.5 normalises that regardless of the input being integer
    (synseg-snap, GT line) or already a half-step float.
    """
    import math

    z = math.floor(float(point_xyz[2])) + 0.5
    ann = {
        "type": "point",
        "id": ann_id,
        "point": [float(point_xyz[0]), float(point_xyz[1]), z],
    }
    if description:
        ann["description"] = description
    return ann


def build_ng_state(
    resolution: Vec3D,
    position: list[float],
    image_path: str,
    pcg_source: str | None,
    seg_path: str | None,
    enriched: dict[str, list[dict]],
) -> dict:
    """Build an NG state with image, optional segmentation, and TP/FP/FN points."""
    dims = {
        "x": [resolution[0] * 1e-9, "m"],
        "y": [resolution[1] * 1e-9, "m"],
        "z": [resolution[2] * 1e-9, "m"],
    }
    layers: list[dict] = [
        {"type": "image", "source": image_path + "/|neuroglancer-precomputed:", "name": "EM"},
    ]
    if pcg_source:
        layers.append(
            {
                "type": "segmentation",
                "source": pcg_source,
                "selectedAlpha": 0.15,
                "segments": [],
                "name": "neuron segmentation",
                "visible": False,
            }
        )
    if seg_path:
        layers.append(
            {
                "type": "segmentation",
                "source": seg_path + "/|neuroglancer-precomputed:",
                "segments": [],
                "name": "Synapse Seg",
            }
        )

    if enriched.get("gt_ignored"):
        layers.append(
            _local_annotation_layer(
                "GT (ignored)",
                enriched["gt_ignored"],
                resolution,
                color="#888888",
                visible=False,
            )
        )
    layers.append(_local_annotation_layer("TP", enriched["tp"], resolution, color="#ffff00"))

    fp_layer = _local_annotation_layer("FP", enriched["fp"], resolution, color="#ff0000")
    # Tagging UI for triaging FPs in the browser.
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
    for a in fp_layer["annotations"]:
        a.setdefault("props", [0, 0, 0])
    layers.append(fp_layer)

    layers.append(_local_annotation_layer("FN", enriched["fn"], resolution, color="#0000ff"))

    if enriched.get("dup_tp"):
        layers.append(
            _local_annotation_layer(
                "Dup Pred (TP)", enriched["dup_tp"], resolution, color="#ff8800"
            )
        )
    if enriched.get("dup_fp"):
        layers.append(
            _local_annotation_layer(
                "Dup Pred (FP)", enriched["dup_fp"], resolution, color="#cc4444"
            )
        )
    if enriched.get("dup_fn_matched"):
        layers.append(
            _local_annotation_layer(
                "Dup GT (matched)", enriched["dup_fn_matched"], resolution, color="#ffffaa"
            )
        )
    if enriched.get("dup_fn_unmatched"):
        layers.append(
            _local_annotation_layer(
                "Dup GT (missed)", enriched["dup_fn_unmatched"], resolution, color="#8888ff"
            )
        )

    return {
        "dimensions": dims,
        "position": position,
        "crossSectionScale": 0.18,
        "projectionScale": 220,
        "layers": layers,
        "selectedLayer": {"layer": "EM", "visible": True},
        "layout": "xy-3d",
    }


def _lookup_keys_at_points(
    pts_vx: list[Vec3D],
    cellseg_path: str,
    resolution: Vec3D,
) -> list[int]:
    """Look up cell IDs at the given (voxel) points by reading a single bbox
    around them from the cellseg layer."""
    if not pts_vx:
        return []
    data, start = load_watershed_region(cellseg_path, pts_vx, resolution)
    return [lookup_sv_id(p, data, start) for p in pts_vx]


def evaluate_detection(
    gt_annotations: list[dict],
    pred_df,
    resolution: Vec3D,
    max_dist_nm: float,
    anchor: str,
    midpoint: bool,
    cellseg_path: str | None = None,
):
    """Hungarian-match GT anchor points to pred centroids by Euclidean nm distance.

    If `cellseg_path` is given, look up cell IDs at the GT anchor / pred centroid
    voxel and apply key-based dedup (matching `eval_synapses.py`). Without
    cellseg, only raw stats are populated.
    """
    gt_pts_vx = [_gt_point(a, anchor, midpoint) for a in gt_annotations]
    pred_pts_vx = [
        Vec3D(float(r.centroid_x), float(r.centroid_y), float(r.centroid_z))
        for _, r in pred_df.iterrows()
    ]
    gt_nm = [p * resolution for p in gt_pts_vx]
    pred_nm = [p * resolution for p in pred_pts_vx]

    stats = match_points(gt_nm, pred_nm, max_dist_nm)

    if cellseg_path:
        # Single bbox read containing all GT + pred points. The lookup helpers
        # in eval_synapses round to nearest voxel and return 0 OOB.
        all_pts = list(gt_pts_vx) + list(pred_pts_vx)
        print(f"  Loading cellseg for {len(all_pts)} points to compute dedup keys...")
        ng = len(gt_pts_vx)
        all_keys = _lookup_keys_at_points(all_pts, cellseg_path, resolution)
        gt_keys = all_keys[:ng]
        pred_keys = all_keys[ng:]
        gt_zero = sum(1 for k in gt_keys if k == 0)
        pred_zero = sum(1 for k in pred_keys if k == 0)
        if gt_zero or pred_zero:
            print(
                f"  WARNING: {gt_zero}/{ng} GT and {pred_zero}/{len(pred_keys)} "
                f"pred points fell on cellseg=0 — those won't dedup."
            )
        _apply_key_dedup(stats, gt_keys, pred_keys, gt_nm, pred_nm, max_dist_nm)
    return stats, gt_pts_vx, pred_pts_vx


def build_enriched(
    stats: dict,
    gt_annotations: list[dict],
    gt_pts_vx: list[Vec3D],
    pred_df,
    pred_pts_vx: list[Vec3D],
    gt_ignored: list[dict],
    anchor: str,
    midpoint: bool,
) -> dict[str, list[dict]]:
    """Convert stats matches into TP/FP/FN/dup_* point annotations."""
    matches = stats["matches"]

    def _pred_desc(j: int) -> str:
        row = pred_df.iloc[j]
        parts = [f"SynID: {int(row.syn_id)}"]
        if hasattr(row, "mean_score") and not pd.isna(row.mean_score):
            parts.append(f"Score: {float(row.mean_score):.4f}")
        return " | ".join(parts)

    def _gt_id(i: int) -> str:
        return str(gt_annotations[i].get("id", uuid.uuid4().hex[:8]))

    # Place TP at the matched GT line's pointA (with the +0.5 z shift applied
    # in _make_point_ann). Description still includes the matched pred's
    # SynID / score.
    tp = [_make_point_ann(gt_pts_vx[gi], f"tp_{_gt_id(gi)}", _pred_desc(pi)) for gi, pi in matches]
    fp = [
        _make_point_ann(pred_pts_vx[j], f"fp_{int(pred_df.iloc[j].syn_id)}", _pred_desc(j))
        for j in stats["fp_pred_indices"]
    ]
    fn = [_make_point_ann(gt_pts_vx[i], f"fn_{_gt_id(i)}") for i in stats["fn_gt_indices"]]
    dup_tp = [
        _make_point_ann(pred_pts_vx[j], f"dup_tp_{int(pred_df.iloc[j].syn_id)}", _pred_desc(j))
        for j in stats.get("dup_tp_pred_indices", [])
    ]
    dup_fp = [
        _make_point_ann(pred_pts_vx[j], f"dup_fp_{int(pred_df.iloc[j].syn_id)}", _pred_desc(j))
        for j in stats.get("dup_fp_pred_indices", [])
    ]
    dup_fn_matched = [
        _make_point_ann(gt_pts_vx[i], f"dup_fn_m_{_gt_id(i)}")
        for i in stats.get("dup_fn_matched_gt_indices", [])
    ]
    dup_fn_unmatched = [
        _make_point_ann(gt_pts_vx[i], f"dup_fn_u_{_gt_id(i)}")
        for i in stats.get("dup_fn_unmatched_gt_indices", [])
    ]

    gt_ignored_anns = [
        _make_point_ann(
            _gt_point(a, anchor, midpoint),
            f"ign_{a.get('id', f'ign_{i}')}",
        )
        for i, a in enumerate(gt_ignored)
    ]
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


def write_outputs(out_dir: str, enriched: dict[str, list[dict]]):
    os.makedirs(out_dir, exist_ok=True)
    for key, anns in enriched.items():
        path = os.path.join(out_dir, f"{key}.json")
        with open(path, "w") as f:
            json.dump(anns, f, indent=2)
        print(f"  Wrote {len(anns)} annotations to {path}")


def parse_args():
    p = argparse.ArgumentParser(description="Ribbon detection evaluation (no assignment).")
    p.add_argument("--gt-json", required=True, help="GT line-annotations JSON.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--synseg-path",
        default=None,
        help="Synseg layer path. Computes CC centroids directly — " "assignment-invariant.",
    )
    src.add_argument(
        "--pred-metadata",
        default=None,
        help="Tabular metadata path (parquet) from the assignment flow. "
        "Detection counts depend on assign_type / merge / size_thr.",
    )
    p.add_argument(
        "--minimum-size",
        type=int,
        default=0,
        help="When using --synseg-path, drop CCs smaller than this many voxels.",
    )
    p.add_argument("--resolution", type=float, nargs=3, default=[16, 16, 40])
    p.add_argument(
        "--max-distance-nm",
        type=float,
        default=600,
        help="Hungarian matching distance threshold in nm.",
    )
    p.add_argument(
        "--gt-anchor",
        choices=["pointA", "pointB", "midpoint"],
        default="pointA",
        help="Which GT line endpoint to use as the detection point. "
        "Ribbons/vesicles → pointA (presyn). PST → pointB (postsyn). "
        "midpoint = average of both ends.",
    )
    p.add_argument(
        "--bbox-start",
        type=int,
        nargs=3,
        default=None,
        help="Optional voxel-coord start; GT/pred outside are dropped.",
    )
    p.add_argument("--bbox-end", type=int, nargs=3, default=None)
    p.add_argument(
        "--output-dir", default=".", help="Local dir for tp/fp/fn JSONs and ng_state.json."
    )
    p.add_argument(
        "--upload-state",
        action="store_true",
        help="Upload an NG state with TP/FP/FN points and print the link.",
    )
    p.add_argument("--image-path", default="gs://stroeh_sem_mouse_retina/image/v2")
    p.add_argument(
        "--pcg-source",
        default=None,
        help="Optional graphene:// or precomputed segmentation source.",
    )
    p.add_argument(
        "--seg-path", default=None, help="Optional precomputed synapse-seg path (e.g. .../synseg)."
    )
    p.add_argument(
        "--cellseg-path",
        default=None,
        help="Optional flat cell-segmentation path. When given, the GT "
        "pointA / pred centroid is mapped to a cell ID and used as a "
        "dedup KEY (e.g. multiple GT lines on the same presyn cell "
        "within --max-distance-nm collapse into one detection).",
    )
    p.add_argument("--cave-server", default="https://global.daf-apis.com")
    p.add_argument("--cave-token", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    resolution: Vec3D = Vec3D(*args.resolution)
    midpoint = args.gt_anchor == "midpoint"
    anchor = args.gt_anchor if args.gt_anchor in ("pointA", "pointB") else "pointA"

    print("Loading ground truth...")
    gt_annotations, gt_res = load_gt(args.gt_json)
    print(f"  {len(gt_annotations)} GT lines at resolution {gt_res}")
    if gt_res != resolution:
        print(f"  WARNING: GT resolution {gt_res} != processing resolution {resolution}")

    print("Loading predictions...")
    if args.synseg_path is not None:
        if args.bbox_start is None or args.bbox_end is None:
            raise SystemExit("--bbox-start and --bbox-end are required with --synseg-path")
        pred_df = predictions_from_synseg(
            args.synseg_path,
            list(args.bbox_start),
            list(args.bbox_end),
            list(args.resolution),
            minimum_size=args.minimum_size,
        )
    else:
        pred_df = load_predictions(args.pred_metadata)
    print(f"  {len(pred_df)} predicted ribbons")

    bbox_st = bbox_end = None
    if args.bbox_start is not None and args.bbox_end is not None:
        bbox_st = Vec3D(*args.bbox_start)
        bbox_end = Vec3D(*args.bbox_end)
        print(f"  Bbox: {bbox_st} -- {bbox_end}")

    pred_df, gt_kept, gt_ignored = _filter_to_bbox(
        pred_df, gt_annotations, bbox_st, bbox_end, anchor, midpoint
    )

    print("Matching...")
    stats, gt_pts_vx, pred_pts_vx = evaluate_detection(
        gt_kept,
        pred_df,
        resolution,
        args.max_distance_nm,
        anchor,
        midpoint,
        cellseg_path=args.cellseg_path,
    )
    print_stats(stats, "RIBBON DETECTION")

    enriched = build_enriched(
        stats,
        gt_kept,
        gt_pts_vx,
        pred_df,
        pred_pts_vx,
        gt_ignored,
        anchor,
        midpoint,
    )
    write_outputs(args.output_dir, enriched)

    if args.upload_state:
        from caveclient import CAVEclient

        cx = (
            float(pred_df["centroid_x"].mean())
            if len(pred_df)
            else float(sum(p[0] for p in gt_pts_vx) / max(len(gt_pts_vx), 1))
        )
        cy = (
            float(pred_df["centroid_y"].mean())
            if len(pred_df)
            else float(sum(p[1] for p in gt_pts_vx) / max(len(gt_pts_vx), 1))
        )
        cz = (
            float(pred_df["centroid_z"].mean())
            if len(pred_df)
            else float(sum(p[2] for p in gt_pts_vx) / max(len(gt_pts_vx), 1))
        )
        position = [cx, cy, cz + 0.5]

        print("\nBuilding Neuroglancer state...")
        ng_state = build_ng_state(
            resolution=resolution,
            position=position,
            image_path=args.image_path,
            pcg_source=args.pcg_source,
            seg_path=args.seg_path,
            enriched=enriched,
        )
        state_path = os.path.join(args.output_dir, "ng_state.json")
        with open(state_path, "w") as f:
            json.dump(ng_state, f, indent=2)
        print(f"  Saved state to {state_path}")

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
