"""
Sweep synapse segmentation threshold to find optimal F1 score.

For each threshold, runs segmentation → assignment → evaluation locally.
Small cutouts only (runs everything in-process).

Supports two modes:
  1. CAVE pipeline (with watershed + ChunkedGraph):
        --watershed-path ... --cave-datastack ... --cave-server ...
  2. Flat segmentation (no CAVE):
        --cellseg-path ... (no --watershed-path)
     Segment IDs are used directly as final cell IDs.

Usage (CAVE):
    python scripts/synapse_evaluation/sweep_threshold.py \
        --gt-json specs/nico/inference/cra9/synapses/gt024_gt_lines.json \
        --pred-path gs://dkronauer-ant-001-synapse/260328_test/gt024_exp0328_50k/20260329152641 \
        --image-path gs://dkronauer-ant-001-alignment-final/aligned \
        --cellseg-path gs://dkronauer-ant-001-segmentations-prod/ng/seg/240904-finetune-v3.2-0.27-size-1x \
        --watershed-path gs://zetta_ws/dkronauer-ant-001-240904-finetune-v3.2-0.27 \
        --model-path gs://dkronauer-ant-001-alex/synapse/experiments/jabae-ant-assign-exp-1119/models/model195000.onnx \
        --output-prefix gs://dkronauer-ant-001-synapse/nkem/cutouts/2026-04-10/gt024_exp0328_50k/sweep

Usage (flat seg):
    python scripts/synapse_evaluation/sweep_threshold.py \
        --gt-json specs/nico/inference/stroeh_retina_redo/synapses/gt_conventional/gt001.json \
        --pred-path gs://zetta-research-nico/stroeh_retina/synapse/eval/pst-exp0301-50k/gt001/predictions \
        --image-path gs://stroeh_sem_mouse_retina/image/v2 \
        --cellseg-path gs://stroeh_sem_mouse_retina_scratch/make_cv_happy/seg/dacey_16-16-40_20250115185710 \
        --model-path gs://alex_research/stroeh_retina/experiments/jabae-stroeh-retina-pst-exp0301/models/model50000.onnx \
        --output-prefix gs://zetta-research-nico/stroeh_retina/synapse/eval/pst-exp0301-50k/gt001/sweep \
        --pred-data-resolution 16 16 40 \
        --resolution 16 16 40
"""

import argparse
import json
import os

from eval_synapses import (
    evaluate,
    load_gt,
    load_predictions,
    print_stats,
)

from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.internal.synapses.syn_assignment import build_synapse_assignment_flow
from zetta_utils.internal.synapses.syn_scores import build_synapse_score_flow
from zetta_utils.internal.synapses.syn_segmentation import (
    build_synapse_segmentation_flow,
)
from zetta_utils.layer.volumetric.cloudvol import build_cv_layer
from zetta_utils.mazepa import execute


def run_segmentation(
    pred_path: str,
    cellseg_path: str,
    dst_path: str,
    bbox: BBox3D,
    resolution: list[float],
    threshold: float,
    pred_data_resolution: list[float] | None = None,
    minimum_size: int = 100,
):
    if pred_data_resolution is None:
        pred_data_resolution = [8, 8, 42]
    flow = build_synapse_segmentation_flow(
        src=build_cv_layer(
            pred_path,
            data_resolution=pred_data_resolution,
            interpolation_mode="img",
        ),
        dst_path=dst_path,
        info_reference_path=cellseg_path,
        bbox=bbox,
        dst_resolution=resolution,
        processing_chunk_sizes=[[256, 256, 64]],
        processing_crop_pads=[[64, 64, 32]],
        threshold=threshold,
        connectivity=26,
        minimum_size=minimum_size,
        segmentation_layer=build_cv_layer(cellseg_path),
        info_chunk_size=[256, 256, 32],
        info_extra_scale_data={"compressed_segmentation_block_size": [16, 16, 4]},
    )
    execute(target=flow, do_dryrun_estimation=False, show_progress=False)


def _bbox_chunk_size(bbox: BBox3D, resolution: list[float]) -> list[int]:
    """Chunk size (in voxels at `resolution`) that exactly covers `bbox`."""
    return [int(round(bbox.shape[i] / resolution[i])) for i in range(3)]


def run_scoring(
    pred_path: str,
    synseg_path: str,
    src_metadata_path: str,
    dst_metadata_path: str,
    bbox: BBox3D,
    resolution: list[float],
    pred_data_resolution: list[float] | None = None,
    processing_chunk_size: list[int] | None = None,
):
    if pred_data_resolution is None:
        pred_data_resolution = list(resolution)
    if processing_chunk_size is None:
        processing_chunk_size = _bbox_chunk_size(bbox, resolution)
    flow = build_synapse_score_flow(
        src_predictions=build_cv_layer(
            pred_path,
            data_resolution=pred_data_resolution,
            interpolation_mode="img",
        ),
        src_synseg=build_cv_layer(synseg_path),
        src_metadata_path=src_metadata_path,
        dst_metadata_path=dst_metadata_path,
        bbox=bbox,
        dst_resolution=resolution,
        processing_chunk_sizes=[processing_chunk_size],
        processing_crop_pads=[[0, 0, 0]],
        dst_metadata_mode="replace",
        expand_bbox_processing=True,
    )
    execute(target=flow, do_dryrun_estimation=False, show_progress=False)


def run_assignment(
    image_path: str,
    synseg_path: str,
    cellseg_path: str,
    watershed_path: str | None,
    model_path: str,
    dst_prefix: str,
    bbox: BBox3D,
    resolution: list[float],
    synapse_type: str = "postsyn",
    assign_type: str = "max",
    assign_thresh: float = 0.0,
    window_size: list[int] | None = None,
    candidate_dilation_xy: int = 2,
    candidate_dilation_z: int = 1,
    processing_chunk_size: list[int] | None = None,
    merge_dist_nm: float = 300,
    size_thr: int = 100,
    crop_pad: list[int] | None = None,
    dilate_max_radius_nm: float = 350.0,
    dilate_max_neighbors: int = 3,
):
    if window_size is None:
        window_size = [24, 24, 8]
    if processing_chunk_size is None:
        processing_chunk_size = _bbox_chunk_size(bbox, resolution)
    # crop_pad serves two purposes: (a) AssignSynapsesOp validates
    # crop_pad * voxel_res >= merge_dist_nm per axis so the merge step has
    # enough cross-chunk overlap, and (b) it gives the assignment net image
    # context around blobs near the chunk edge — without it, edge blobs see
    # truncated windows, get poor scores, and drop out at dedup. Default
    # matches the in-repo cue specs (inference_assignment.cue) at 16nm xy.
    if crop_pad is None:
        crop_pad = [96, 96, 16]
    src_watershed = build_cv_layer(watershed_path) if watershed_path else None
    flow = build_synapse_assignment_flow(
        src_image=build_cv_layer(image_path),
        src_synseg=build_cv_layer(synseg_path),
        src_cellseg=build_cv_layer(cellseg_path),
        src_watershed=src_watershed,
        dst=build_cv_layer(
            dst_prefix + "/seg",
            info_reference_path=synseg_path,
            info_inherit_all_params=True,
            info_scales=[resolution],
            # Match dst storage chunks to processing chunks. Otherwise a
            # processing_chunk that doesn't align with the synseg's storage
            # grid (256x256x32) hits assert_idx_is_chunk_aligned on write.
            info_chunk_size=processing_chunk_size,
            info_overwrite=True,
        ),
        dst_metadata_path=dst_prefix + "/metadata",
        dst_metadata_mode="replace",
        bbox=bbox,
        dst_resolution=resolution,
        processing_chunk_sizes=[processing_chunk_size],
        processing_crop_pads=[crop_pad],
        expand_bbox_processing=True,
        model_path=model_path,
        window_size=window_size,
        merge_dist_nm=merge_dist_nm,
        size_thr=size_thr,
        synapse_type=synapse_type,
        assign_type=assign_type,
        assign_thresh=assign_thresh,
        candidate_dilation_xy=candidate_dilation_xy,
        candidate_dilation_z=candidate_dilation_z,
        dilate_max_radius_nm=dilate_max_radius_nm,
        dilate_max_neighbors=dilate_max_neighbors,
    )
    execute(target=flow, do_dryrun_estimation=False, show_progress=False)


def parse_args():
    p = argparse.ArgumentParser(description="Sweep segmentation threshold for best F1")
    p.add_argument("--gt-json", required=True)
    p.add_argument("--pred-path", required=True, help="GCS path to raw PST predictions")
    p.add_argument("--image-path", required=True)
    p.add_argument("--cellseg-path", required=True)
    p.add_argument(
        "--watershed-path", default=None, help="GCS path to watershed (CAVE pipeline only)"
    )
    p.add_argument(
        "--model-path",
        default=None,
        help="GCS path to the assignment ONNX. Required for --assign-type "
        "max/pre_thresh/post_thresh; ignored for --assign-type dilate_nearest.",
    )
    p.add_argument(
        "--bbox-start",
        type=int,
        nargs=3,
        default=None,
        help="Start coord in voxels at --resolution (default: derive from pred layer bounds)",
    )
    p.add_argument(
        "--bbox-end",
        type=int,
        nargs=3,
        default=None,
        help="End coord in voxels at --resolution (default: derive from pred layer bounds)",
    )
    p.add_argument("--output-prefix", required=True, help="GCS prefix for sweep outputs")
    p.add_argument("--resolution", type=float, nargs=3, default=[16, 16, 42])
    p.add_argument(
        "--pred-data-resolution",
        type=float,
        nargs=3,
        default=None,
        help="Native resolution of prediction layer (default: same as --resolution)",
    )
    p.add_argument("--max-distance-nm", type=float, default=600)
    p.add_argument(
        "--cave-datastack", default=None, help="CAVE datastack (required with --watershed-path)"
    )
    p.add_argument("--cave-server", default="https://proofreading.zetta.ai")
    p.add_argument("--cave-token", default=None)
    p.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[
            0.005,
            0.010,
            0.015,
            0.020,
            0.025,
            0.030,
            0.035,
            0.040,
            0.045,
            0.050,
            0.055,
            0.060,
        ],
    )
    p.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Beta for F-beta scoring used to pick the best threshold. "
        "beta>1 weights recall higher than precision (e.g. beta=2 when "
        "many predictions are likely true synapses missing from GT).",
    )
    p.add_argument(
        "--minimum-size",
        type=int,
        default=100,
        help="Minimum connected-component size (voxels) in the synseg flow.",
    )
    p.add_argument(
        "--size-thr",
        type=int,
        default=100,
        help="Size threshold (voxels) inside AssignSynapsesOp; small synseg blobs "
        "are dropped before assignment.",
    )
    p.add_argument(
        "--merge-dist-nm",
        type=float,
        default=300.0,
        help="Merge distance for AssignSynapsesOp (nm).",
    )
    p.add_argument(
        "--assignment-crop-pad",
        type=int,
        nargs=3,
        default=None,
        metavar=("PX", "PY", "PZ"),
        help="Override the assignment flow's processing_crop_pads (voxels at "
        "--resolution). Default: [96, 96, 16] — matches inference_assignment.cue. "
        "Don't drop below this without checking edge-blob scoring: the assignment "
        "net needs image context beyond the bbox to score blobs near the edge.",
    )
    p.add_argument(
        "--assignment-chunk-size",
        type=int,
        nargs=3,
        default=None,
        metavar=("CX", "CY", "CZ"),
        help="Override the assignment flow's processing_chunk_size (voxels). "
        "Default: full bbox (one chunk). Set this to a smaller value when "
        "the bbox is too large to fit in RAM (image + cellseg + watershed "
        "+ synseg load at full resolution).",
    )
    p.add_argument(
        "--boundary-margin",
        type=int,
        nargs=3,
        default=None,
        metavar=("MX", "MY", "MZ"),
        help="Exclude predictions with centroid within this many voxels of bbox edge from FP",
    )
    p.add_argument("--output-dir", default=".", help="Local dir for result JSONs and summary")
    p.add_argument(
        "--synapse-type",
        choices=["cleft", "postsyn", "presyn"],
        default="postsyn",
        help="Which side the synapse segmentation represents (cleft, postsyn=PST, presyn=vesicle/ribbon)",
    )
    p.add_argument(
        "--assign-type",
        choices=[
            "max",
            "pre_thresh",
            "post_thresh",
            "pre_thresh_or_max",
            "post_thresh_or_max",
            "dilate_nearest",
        ],
        default="max",
        help="How to pick partner cell(s). 'max' = single best; 'post_thresh'/'pre_thresh' = "
        "all cells over --assign-thresh (for ribbons or other multi-partner synapses); "
        "'pre_thresh_or_max'/'post_thresh_or_max' = like *_thresh but ALSO keep the "
        "argmax cell unconditionally, so every PST blob gets at least one partner "
        "even when no cell exceeds the threshold; "
        "'dilate_nearest' = no-network baseline that picks the closest N cells in nm-space.",
    )
    p.add_argument(
        "--assign-thresh",
        type=float,
        default=0.0,
        help="Threshold for --assign-type=post_thresh/pre_thresh",
    )
    p.add_argument(
        "--dilate-max-radius-nm",
        type=float,
        default=350.0,
        help="Maximum search radius (nm) for --assign-type=dilate_nearest.",
    )
    p.add_argument(
        "--dilate-max-neighbors",
        type=int,
        default=3,
        help="Maximum number of partner cells for --assign-type=dilate_nearest.",
    )
    p.add_argument(
        "--window-size",
        type=int,
        nargs=3,
        default=[24, 24, 8],
        metavar=("X", "Y", "Z"),
        help="Window size (in voxels) for the assignment model. Must match the "
        "shape the assignment ONNX expects. PST/vesicle: [24, 24, 8]. Ribbon: [64, 64, 8].",
    )
    p.add_argument(
        "--candidate-dilation-xy",
        type=int,
        default=2,
        help="XY dilation (in voxels) for candidate cell search. Default 2 works "
        "for PST/vesicle masks near the cleft. Ribbons sit deeper inside the "
        "presyn terminal; bump to 5-10 to reach across the cleft.",
    )
    p.add_argument(
        "--candidate-dilation-z",
        type=int,
        default=1,
        help="Z dilation (in voxels) for candidate cell search (default 1).",
    )
    p.add_argument(
        "--skip-segmentation",
        action="store_true",
        help="Reuse existing synseg at output-prefix/thr_<T>/synseg and only rerun "
        "assignment + scoring + eval. Useful when tuning assignment-side params.",
    )
    p.add_argument(
        "--skip-assignment",
        action="store_true",
        help="Reuse existing assignment metadata at output-prefix/thr_<T>/assignment/"
        "metadata. Implies --skip-segmentation. Useful when only tuning scoring "
        "or eval params.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.assign_type != "dilate_nearest" and not args.model_path:
        raise SystemExit(
            f"--model-path is required for --assign-type={args.assign_type!r} "
            "(only 'dilate_nearest' runs without a network)."
        )
    resolution = list(args.resolution)
    pred_data_resolution = (
        list(args.pred_data_resolution) if args.pred_data_resolution else list(resolution)
    )
    res_vec = Vec3D(*resolution)

    # Derive bbox from pred layer bounds if not explicitly provided.
    user_bbox = args.bbox_start is not None and args.bbox_end is not None
    if not user_bbox:
        print(f"Deriving bbox from pred layer bounds: {args.pred_path}")
        pred_res = Vec3D(*pred_data_resolution)
        pred_layer = build_cv_layer(args.pred_path)
        bounds = pred_layer.backend.get_bounds(pred_res)
        slices = bounds.to_slices()
        # Convert from pred resolution to processing resolution
        scale = [pred_data_resolution[i] / resolution[i] for i in range(3)]
        bbox_start = [int(int(slices[i].start) * scale[i]) for i in range(3)]
        bbox_end = [int(int(slices[i].stop) * scale[i]) for i in range(3)]
        print(f"  Derived bbox at {tuple(resolution)}nm: {bbox_start} -- {bbox_end}")
    else:
        bbox_start = list(args.bbox_start)
        bbox_end = list(args.bbox_end)

    # When the bbox is auto-derived, expand by 1 voxel on each face so that
    # synapses touching the evaluation bbox edge are captured in full
    # (CCEdgeClear preserves them but truncated slivers may fall below min_size).
    # When the user supplied --bbox-start/--bbox-end explicitly, take those
    # bounds verbatim — the user has likely specified the exact cutout bbox
    # to match a prior inference run, and any padding would shift counts.
    pad = 0 if user_bbox else 1
    bbox = BBox3D.from_coords(
        start_coord=[bbox_start[i] - pad for i in range(3)],
        end_coord=[bbox_end[i] + pad for i in range(3)],
        resolution=resolution,
    )

    print("Loading ground truth...")
    gt_annotations, gt_res = load_gt(args.gt_json)
    print(f"  {len(gt_annotations)} GT lines at resolution {gt_res}")

    cave_client = None
    if args.watershed_path:
        from caveclient import CAVEclient

        if not args.cave_datastack:
            raise ValueError("--cave-datastack is required when using --watershed-path")
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

    results = []

    for thr in args.thresholds:
        print(f"\n{'='*60}")
        print(f"  THRESHOLD = {thr}")
        print(f"{'='*60}")

        synseg_path = f"{args.output_prefix}/thr_{thr:.4f}/synseg"
        assign_prefix = f"{args.output_prefix}/thr_{thr:.4f}/assignment"
        metadata_path = assign_prefix + "/metadata"
        metadata_scored_path = assign_prefix + "/metadata_scored"

        if args.skip_segmentation:
            print(f"  Reusing existing segmentation at {synseg_path}")
        else:
            print("  Running segmentation...")
            run_segmentation(
                args.pred_path,
                args.cellseg_path,
                synseg_path,
                bbox,
                resolution,
                thr,
                pred_data_resolution=pred_data_resolution,
                minimum_size=args.minimum_size,
            )

        if args.skip_assignment:
            print(f"  Reusing existing assignment metadata at {metadata_path}")
        else:
            print("  Running assignment...")
            run_assignment(
                args.image_path,
                synseg_path,
                args.cellseg_path,
                args.watershed_path,
                args.model_path or "",
                assign_prefix,
                bbox,
                resolution,
                synapse_type=args.synapse_type,
                assign_type=args.assign_type,
                assign_thresh=args.assign_thresh,
                window_size=list(args.window_size),
                candidate_dilation_xy=args.candidate_dilation_xy,
                candidate_dilation_z=args.candidate_dilation_z,
                merge_dist_nm=args.merge_dist_nm,
                size_thr=args.size_thr,
                crop_pad=list(args.assignment_crop_pad) if args.assignment_crop_pad else None,
                processing_chunk_size=list(args.assignment_chunk_size)
                if args.assignment_chunk_size
                else None,
                dilate_max_radius_nm=args.dilate_max_radius_nm,
                dilate_max_neighbors=args.dilate_max_neighbors,
            )

        print("  Running scoring...")
        run_scoring(
            args.pred_path,
            synseg_path,
            metadata_path,
            metadata_scored_path,
            bbox,
            resolution,
            pred_data_resolution=pred_data_resolution,
            # Match scoring's chunking to the assignment's so scoring can
            # read the parquet chunk-by-chunk (the source layer's chunk_size
            # was set by the assignment flow).
            processing_chunk_size=list(args.assignment_chunk_size)
            if args.assignment_chunk_size
            else None,
        )

        print("  Loading predictions...")
        pred_df = load_predictions(metadata_scored_path)
        print(f"    {len(pred_df)} predicted assignments")

        print("  Evaluating...")
        detect_stats, assign_stats, context = evaluate(
            gt_annotations,
            pred_df,
            args.watershed_path,
            cave_client,
            res_vec,
            args.max_distance_nm,
            boundary_margin=tuple(args.boundary_margin) if args.boundary_margin else None,
            metadata_path=metadata_scored_path,
            segmentation_path=None if args.watershed_path else args.cellseg_path,
            synapse_type=args.synapse_type,
        )
        print_stats(detect_stats, f"  DETECTION ACCURACY (thr={thr})")
        print_stats(assign_stats, f"  ASSIGNMENT ACCURACY (thr={thr})")

        results.append(
            {
                "threshold": thr,
                "n_gt": len(context["gt_annotations"]),
                "n_pred": len(pred_df),
                "detect_tp": detect_stats["tp"],
                "detect_fp": detect_stats["fp"],
                "detect_fn": detect_stats["fn"],
                "detect_precision": detect_stats["precision"],
                "detect_recall": detect_stats["recall"],
                "detect_f1": detect_stats["f1"],
                "detect_dup_tp": detect_stats.get("dup_tp", 0),
                "detect_dup_fp": detect_stats.get("dup_fp", 0),
                "detect_dup_fn": detect_stats.get("dup_fn", 0),
                "detect_precision_dedup": detect_stats.get(
                    "precision_dedup", detect_stats["precision"]
                ),
                "detect_recall_dedup": detect_stats.get("recall_dedup", detect_stats["recall"]),
                "detect_f1_dedup": detect_stats.get("f1_dedup", detect_stats["f1"]),
                "assign_tp": assign_stats["tp"],
                "assign_fp": assign_stats["fp"],
                "assign_fn": assign_stats["fn"],
                "assign_precision": assign_stats["precision"],
                "assign_recall": assign_stats["recall"],
                "assign_f1": assign_stats["f1"],
                "assign_dup_tp": assign_stats.get("dup_tp", 0),
                "assign_dup_fp": assign_stats.get("dup_fp", 0),
                "assign_dup_fn": assign_stats.get("dup_fn", 0),
                "assign_precision_dedup": assign_stats.get(
                    "precision_dedup", assign_stats["precision"]
                ),
                "assign_recall_dedup": assign_stats.get("recall_dedup", assign_stats["recall"]),
                "assign_f1_dedup": assign_stats.get("f1_dedup", assign_stats["f1"]),
            }
        )

    # --- Summary ---
    # Raw  = Hungarian 1:1 match (per GT line / per pred row)
    # Dedup = duplicates (same key within max-distance of TP) reclassified
    beta = args.beta
    b2 = beta * beta

    def fbeta(p: float, r: float) -> float:
        denom = b2 * p + r
        return (1 + b2) * p * r / denom if denom > 0 else 0.0

    for r in results:
        r["detect_fbeta_dedup"] = fbeta(r["detect_precision_dedup"], r["detect_recall_dedup"])
        r["assign_fbeta_dedup"] = fbeta(r["assign_precision_dedup"], r["assign_recall_dedup"])

    width = 130
    fb_label = f"F{beta:g}".upper().replace("F1", "F1 ")
    print(f"\n{'='*width}")
    print(
        f"SWEEP SUMMARY (raw = per-row Hungarian, dedup = same-key-nearby reclassified, picking by F{beta:g})"
    )
    print(f"{'='*width}")
    print(
        f"{'thr':>7} {'#pred':>6} | "
        f"{'DET_P':>6} {'DET_R':>6} {'DET_F1':>6} {'dDT_P':>6} {'dDT_R':>6} {'dDT_F1':>6} {'dDT_'+fb_label:>7} | "
        f"{'ASN_P':>6} {'ASN_R':>6} {'ASN_F1':>6} {'dAS_P':>6} {'dAS_R':>6} {'dAS_F1':>6} {'dAS_'+fb_label:>7}"
    )
    print("-" * width)
    best_detect = max(results, key=lambda r: r["detect_fbeta_dedup"])
    best_assign = max(results, key=lambda r: r["assign_fbeta_dedup"])
    for r in results:
        marker = ""
        if r["threshold"] == best_detect["threshold"]:
            marker += " <- best DET"
        if r["threshold"] == best_assign["threshold"]:
            marker += " <- best ASN"
        print(
            f"{r['threshold']:>7.4f} {r['n_pred']:>6} | "
            f"{r['detect_precision']:>6.3f} {r['detect_recall']:>6.3f} {r['detect_f1']:>6.3f} "
            f"{r['detect_precision_dedup']:>6.3f} {r['detect_recall_dedup']:>6.3f} {r['detect_f1_dedup']:>6.3f} "
            f"{r['detect_fbeta_dedup']:>7.3f} | "
            f"{r['assign_precision']:>6.3f} {r['assign_recall']:>6.3f} {r['assign_f1']:>6.3f} "
            f"{r['assign_precision_dedup']:>6.3f} {r['assign_recall_dedup']:>6.3f} {r['assign_f1_dedup']:>6.3f} "
            f"{r['assign_fbeta_dedup']:>7.3f}"
            f"{marker}"
        )
    print(
        f"\nBest Detection F{beta:g} (dedup):  {best_detect['detect_fbeta_dedup']:.3f} at threshold {best_detect['threshold']}"
    )
    print(
        f"Best Assignment F{beta:g} (dedup): {best_assign['assign_fbeta_dedup']:.3f} at threshold {best_assign['threshold']}"
    )

    # Write best-threshold eval outputs
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "sweep_results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved sweep results to {summary_path}")

    # Print final eval command for the best assignment-Fbeta threshold
    best_thr = best_assign["threshold"]
    best_metadata = f"{args.output_prefix}/thr_{best_thr:.4f}/assignment/metadata_scored"
    print(f"\n{'='*96}")
    print(f"Final eval command for best threshold ({best_thr}):")
    print(f"{'='*96}")
    margin_str = ""
    if args.boundary_margin:
        mx, my, mz = args.boundary_margin
        margin_str = f" \\\n  --boundary-margin {mx} {my} {mz}"
    if args.watershed_path:
        seg_str = (
            f"  --watershed-path {args.watershed_path} \\\n"
            f"  --cave-datastack {args.cave_datastack} \\\n"
        )
    else:
        seg_str = f"  --segmentation-path {args.cellseg_path} \\\n"
    print(
        f"python scripts/synapse_evaluation/eval_synapses.py \\\n"
        f"  --gt-json {args.gt_json} \\\n"
        f"  --pred-metadata {best_metadata} \\\n"
        f"{seg_str}"
        f"  --synapse-type {args.synapse_type} \\\n"
        f"  --output-dir {args.output_dir}"
        f"{margin_str}"
    )


if __name__ == "__main__":
    main()
