#!/usr/bin/env python3
"""Compute aggregate metrics for contact merge predictions across all chunks.

Reads written merge_probabilities and compares to ground truth merge_decisions.
Computes AUC-PR, AUC-ROC, precision, recall for ALL and AMBIGUOUS (MA 0.15-0.35) samples.
Optionally saves PR and ROC curves as PDF.

Usage:
    python compute_merge_metrics.py \
        --source-path gs://dodam_exp/fafb_v14/seg_contacts/fafb_v14_cutout_x2_contacts_x1 \
        --path gs://martin_exp/contact_merge_inference_test/fafb_v14_cutout_x2_contacts_x1 \
        --authority model_v6.0_r2k_p2k_pn4k_cf_aff_bs64_minbs16_lr1e-3_test \
        --gt-authority ground_truth \
        --bbox 29664,9364,1240,34808,12460,1304 \
        --output-dir ./curves
"""

import argparse
import json
import os
import struct
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from google.cloud import storage
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from tqdm import tqdm


def parse_bbox(bbox_str: str) -> tuple[list[int], list[int]]:
    """Parse bbox string 'x0,y0,z0,x1,y1,z1' into start and end coords."""
    coords = [int(x) for x in bbox_str.split(",")]
    return coords[:3], coords[3:]


def parse_vec(vec_str: str) -> list[int]:
    """Parse vector string 'x,y,z' into list."""
    return [int(x) for x in vec_str.split(",")]


def read_info(bucket, base_path: str) -> dict:
    """Read info file from seg_contact layer."""
    blob = bucket.blob(os.path.join(base_path, "info"))
    if not blob.exists():
        raise FileNotFoundError(f"Info file not found: {os.path.join(base_path, 'info')}")
    return json.loads(blob.download_as_bytes().decode("utf-8"))


def get_chunk_keys_from_info(
    info: dict, bbox_start: list[int], bbox_end: list[int]
) -> list[str]:
    """Generate chunk keys from info file grid, filtered by bbox overlap.

    Uses the backend's naming convention (full chunk_size, no clipping at edges).
    """
    voxel_offset = info["voxel_offset"]
    size = info["size"]
    chunk_size = info["chunk_size"]

    keys = []
    for x in range(voxel_offset[0], voxel_offset[0] + size[0], chunk_size[0]):
        x_end = x + chunk_size[0]
        if x_end <= bbox_start[0] or x >= bbox_end[0]:
            continue
        for y in range(voxel_offset[1], voxel_offset[1] + size[1], chunk_size[1]):
            y_end = y + chunk_size[1]
            if y_end <= bbox_start[1] or y >= bbox_end[1]:
                continue
            for z in range(voxel_offset[2], voxel_offset[2] + size[2], chunk_size[2]):
                z_end = z + chunk_size[2]
                if z_end <= bbox_start[2] or z >= bbox_end[2]:
                    continue
                keys.append(f"{x}-{x_end}_{y}-{y_end}_{z}-{z_end}")
    return keys


def read_contacts_chunk(bucket, path: str) -> dict[int, dict]:
    """Read contacts from a chunk, return dict keyed by contact_id."""
    blob = bucket.blob(path)
    if not blob.exists():
        return {}

    data = blob.download_as_bytes()
    contacts = {}

    with io.BytesIO(data) as f:
        n_contacts = struct.unpack("<I", f.read(4))[0]
        for _ in range(n_contacts):
            contact_id = struct.unpack("<q", f.read(8))[0]
            seg_a, seg_b = struct.unpack("<qq", f.read(16))
            com = struct.unpack("<fff", f.read(12))
            n_faces = struct.unpack("<I", f.read(4))[0]

            # Read contact faces (xyz + affinity)
            faces = []
            for _ in range(n_faces):
                face = struct.unpack("<ffff", f.read(16))
                faces.append(face)

            # Read metadata
            metadata_len = struct.unpack("<I", f.read(4))[0]
            if metadata_len > 0:
                f.read(metadata_len)

            # Compute mean affinity using pytorch to match training pipeline
            if faces:
                faces_tensor = torch.tensor(faces, dtype=torch.float32)
                affinities = faces_tensor[:, 3]
                nonzero_mask = (
                    (faces_tensor[:, 0] != 0)
                    | (faces_tensor[:, 1] != 0)
                    | (faces_tensor[:, 2] != 0)
                )
                nonzero_count = nonzero_mask.sum().clamp(min=1)
                mean_affinity = ((affinities * nonzero_mask).sum() / nonzero_count).item()
            else:
                mean_affinity = 0.0

            contacts[contact_id] = {
                "seg_a": seg_a,
                "seg_b": seg_b,
                "com": com,
                "mean_affinity": mean_affinity,
            }

    return contacts


def read_merge_decisions_chunk(bucket, path: str) -> dict[int, bool]:
    """Read merge decisions from a chunk, return dict keyed by contact_id."""
    blob = bucket.blob(path)
    if not blob.exists():
        return {}

    data = blob.download_as_bytes()
    decisions = {}

    with io.BytesIO(data) as f:
        n_decisions = struct.unpack("<I", f.read(4))[0]
        for _ in range(n_decisions):
            contact_id = struct.unpack("<q", f.read(8))[0]
            should_merge = struct.unpack("<B", f.read(1))[0]
            decisions[contact_id] = bool(should_merge)

    return decisions


def read_merge_probabilities_chunk(bucket, path: str) -> dict[int, float]:
    """Read merge probabilities from a chunk, return dict keyed by contact_id."""
    blob = bucket.blob(path)
    if not blob.exists():
        return {}

    data = blob.download_as_bytes()
    probs = {}

    with io.BytesIO(data) as f:
        n_entries = struct.unpack("<I", f.read(4))[0]
        for _ in range(n_entries):
            contact_id = struct.unpack("<q", f.read(8))[0]
            prob = struct.unpack("<f", f.read(4))[0]
            probs[contact_id] = prob

    return probs


def plot_curves(
    targets: np.ndarray,
    probs: np.ndarray,
    mean_affinities: np.ndarray,
    subset_name: str,
    output_dir: Path,
    authority: str,
) -> None:
    """Plot and save PR and ROC curves as PDF."""
    # Compute curves for classifier
    pr_precision, pr_recall, _ = precision_recall_curve(targets, probs)
    auc_pr_classifier = auc(pr_recall, pr_precision)
    fpr, tpr, _ = roc_curve(targets, probs)
    auc_roc_classifier = auc(fpr, tpr)

    # Compute curves for mean affinity baseline
    pr_ma_precision, pr_ma_recall, _ = precision_recall_curve(targets, mean_affinities)
    auc_pr_ma = auc(pr_ma_recall, pr_ma_precision)
    fpr_ma, tpr_ma, _ = roc_curve(targets, mean_affinities)
    auc_roc_ma = auc(fpr_ma, tpr_ma)

    # Sanitize subset name for filename
    safe_subset = subset_name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")

    # PR Curve
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(pr_recall, pr_precision, label=f"Classifier (AUC={auc_pr_classifier:.3f})", linewidth=2)
    ax.plot(pr_ma_recall, pr_ma_precision, label=f"Mean Affinity (AUC={auc_pr_ma:.3f})", linewidth=2, linestyle="--")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"PR Curve - {subset_name}", fontsize=14)
    ax.legend(loc="lower left", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    pr_path = output_dir / f"pr_curve_{safe_subset}_{authority}.pdf"
    fig.savefig(pr_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {pr_path}")

    # ROC Curve
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"Classifier (AUC={auc_roc_classifier:.3f})", linewidth=2)
    ax.plot(fpr_ma, tpr_ma, label=f"Mean Affinity (AUC={auc_roc_ma:.3f})", linewidth=2, linestyle="--")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1)  # Random baseline
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve - {subset_name}", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    roc_path = output_dir / f"roc_curve_{safe_subset}_{authority}.pdf"
    fig.savefig(roc_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {roc_path}")


def compute_metrics(
    targets: np.ndarray,
    probs: np.ndarray,
    mean_affinities: np.ndarray,
    subset_name: str,
    output_dir: Path | None = None,
    authority: str = "",
) -> dict:
    """Compute and print metrics for a subset."""
    n_samples = len(targets)
    n_positive = int(targets.sum())
    n_negative = n_samples - n_positive

    # Predictions at threshold 0.5
    preds = (probs > 0.5).astype(float)
    tp = ((preds == 1) & (targets == 1)).sum()
    fp = ((preds == 1) & (targets == 0)).sum()
    fn = ((preds == 0) & (targets == 1)).sum()
    tn = ((preds == 0) & (targets == 0)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    accuracy = (preds == targets).mean()

    # PR curve and AUC
    pr_precision, pr_recall, _ = precision_recall_curve(targets, probs)
    auc_pr_classifier = auc(pr_recall, pr_precision)

    # ROC curve and AUC
    fpr, tpr, _ = roc_curve(targets, probs)
    auc_roc_classifier = auc(fpr, tpr)

    # Mean affinity as baseline
    pr_ma_precision, pr_ma_recall, _ = precision_recall_curve(targets, mean_affinities)
    auc_pr_ma = auc(pr_ma_recall, pr_ma_precision)
    fpr_ma, tpr_ma, _ = roc_curve(targets, mean_affinities)
    auc_roc_ma = auc(fpr_ma, tpr_ma)

    print(f"\n{'='*60}")
    print(f"METRICS: {subset_name}")
    print(f"{'='*60}")
    print(f"  Samples: {n_samples} (pos={n_positive}, neg={n_negative})")
    print(f"  Confusion: TP={int(tp)}, FP={int(fp)}, FN={int(fn)}, TN={int(tn)}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  ")
    print(f"  AUC-PR  Classifier: {auc_pr_classifier:.4f}  MeanAffinity: {auc_pr_ma:.4f}")
    print(f"  AUC-ROC Classifier: {auc_roc_classifier:.4f}  MeanAffinity: {auc_roc_ma:.4f}")

    # Save curves if output_dir provided
    if output_dir is not None:
        plot_curves(targets, probs, mean_affinities, subset_name, output_dir, authority)

    return {
        "n_samples": n_samples,
        "n_positive": n_positive,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "auc_pr_classifier": auc_pr_classifier,
        "auc_pr_ma": auc_pr_ma,
        "auc_roc_classifier": auc_roc_classifier,
        "auc_roc_ma": auc_roc_ma,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute aggregate merge metrics")
    parser.add_argument("--path", required=True, help="Seg_contact path with predictions (destination)")
    parser.add_argument("--source-path", required=True, help="Source seg_contact path for contacts and GT")
    parser.add_argument("--authority", required=True, help="Merge probability authority name")
    parser.add_argument("--gt-authority", default="ground_truth", help="Ground truth authority name")
    parser.add_argument("--bbox", required=True, help="Bounding box: x0,y0,z0,x1,y1,z1")
    parser.add_argument("--resolution", default="16,16,40", help="Resolution: x,y,z")
    parser.add_argument("--min-mean-affinity", type=float, default=0.1, help="Min mean affinity filter")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save PR/ROC curve PDFs")
    args = parser.parse_args()

    # Setup output directory if specified
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")

    start, end = parse_bbox(args.bbox)
    resolution = parse_vec(args.resolution)

    # Parse prediction (destination) path
    pred_bucket_name = args.path.replace("gs://", "").split("/")[0]
    pred_base_path = "/".join(args.path.replace("gs://", "").split("/")[1:])

    # Parse source path (for contacts and GT)
    source_path = args.source_path
    src_bucket_name = source_path.replace("gs://", "").split("/")[0]
    src_base_path = "/".join(source_path.replace("gs://", "").split("/")[1:])

    client = storage.Client()
    src_bucket = client.bucket(src_bucket_name)
    pred_bucket = client.bucket(pred_bucket_name)

    # Validate that predictions exist
    pred_prob_prefix = os.path.join(pred_base_path, "merge_probabilities", args.authority)
    if not list(pred_bucket.list_blobs(prefix=pred_prob_prefix, max_results=1)):
        raise FileNotFoundError(
            f"No predictions found at gs://{pred_bucket_name}/{pred_prob_prefix}. "
            f"Check --path and --authority."
        )

    # Read info file from SOURCE and generate chunk keys from the actual data grid
    info = read_info(src_bucket, src_base_path)
    chunk_size = info["chunk_size"]
    chunk_keys = get_chunk_keys_from_info(info, start, end)

    # COM bounds in nm for filtering (matches backend's start_nm <= com < end_nm)
    bbox_start_nm = [start[i] * resolution[i] for i in range(3)]
    bbox_end_nm = [end[i] * resolution[i] for i in range(3)]

    print(f"Source path: {source_path}")
    print(f"Predictions path: {args.path}")
    print(f"Authority: {args.authority}")
    print(f"GT Authority: {args.gt_authority}")
    print(f"BBox: {start} -> {end}")
    print(f"Chunk size (from info): {chunk_size}")
    print(f"Total chunks: {len(chunk_keys)}")
    print(f"Min mean affinity: {args.min_mean_affinity}")

    # Collect data from all chunks
    all_targets = []
    all_probs = []
    all_mean_affinities = []

    n_chunks_with_data = 0
    n_contacts_total = 0
    n_with_gt = 0
    n_with_pred = 0
    n_in_bbox = 0
    n_no_gt = 0
    n_no_pred = 0
    n_below_affinity = 0
    n_matched = 0
    n_pred_not_in_source = 0

    for chunk_key in tqdm(chunk_keys, desc="Reading chunks"):
        # Read contacts and GT from SOURCE (original, unmodified contact_faces)
        contacts = read_contacts_chunk(
            src_bucket, os.path.join(src_base_path, "contacts", chunk_key)
        )

        gt_decisions = read_merge_decisions_chunk(
            src_bucket, os.path.join(src_base_path, "merge_decisions", args.gt_authority, chunk_key)
        )

        # Read predictions from DESTINATION (inference output)
        predictions = read_merge_probabilities_chunk(
            pred_bucket, os.path.join(pred_base_path, "merge_probabilities", args.authority, chunk_key)
        )

        if not contacts:
            continue

        n_chunks_with_data += 1
        n_contacts_total += len(contacts)
        n_with_gt += len(gt_decisions)
        n_with_pred += len(predictions)

        # Count predictions not in source (stale/orphan predictions from previous runs)
        n_pred_not_in_source += sum(1 for pid in predictions if pid not in contacts)

        # Match contacts with both GT and predictions
        for contact_id, contact in contacts.items():
            # Filter by COM within bbox (matches backend's start_nm <= com < end_nm)
            com = contact["com"]
            if not (
                bbox_start_nm[0] <= com[0] < bbox_end_nm[0]
                and bbox_start_nm[1] <= com[1] < bbox_end_nm[1]
                and bbox_start_nm[2] <= com[2] < bbox_end_nm[2]
            ):
                continue

            n_in_bbox += 1

            if contact_id not in gt_decisions:
                n_no_gt += 1
                continue
            if contact_id not in predictions:
                n_no_pred += 1
                continue

            mean_aff = contact["mean_affinity"]

            if mean_aff < args.min_mean_affinity:
                n_below_affinity += 1
                continue

            n_matched += 1
            all_targets.append(1.0 if gt_decisions[contact_id] else 0.0)
            all_probs.append(predictions[contact_id])
            all_mean_affinities.append(mean_aff)


    print(f"\nData collection:")
    print(f"  Chunks with data: {n_chunks_with_data}/{len(chunk_keys)}")
    print(f"  Total contacts (source): {n_contacts_total}")
    print(f"  With ground truth (source): {n_with_gt}")
    print(f"  With predictions (dest): {n_with_pred}")
    print(f"  Filter breakdown (contacts in bbox: {n_in_bbox}):")
    print(f"    - No GT:              {n_no_gt}")
    print(f"    - No prediction:      {n_no_pred}")
    print(f"    - Below min affinity: {n_below_affinity}")
    print(f"    - Matched:            {n_matched}")
    if n_pred_not_in_source > 0:
        print(f"  WARNING: {n_pred_not_in_source} predictions in dest have no matching contact in source (stale data?)")

    if n_matched < 10:
        print("\nERROR: Not enough matched samples for metrics computation")
        return

    targets = np.array(all_targets)
    probs = np.array(all_probs)
    mean_affinities = np.array(all_mean_affinities)

    # Compute metrics for ALL samples
    compute_metrics(targets, probs, mean_affinities, "ALL", output_dir, args.authority)

    # Compute metrics for AMBIGUOUS samples (MA 0.15-0.35)
    ambig_mask = (mean_affinities >= 0.15) & (mean_affinities <= 0.35)
    n_ambig = ambig_mask.sum()
    if n_ambig >= 10:
        compute_metrics(
            targets[ambig_mask],
            probs[ambig_mask],
            mean_affinities[ambig_mask],
            f"AMBIGUOUS (MA 0.15-0.35, n={n_ambig})",
            output_dir,
            args.authority,
        )
    else:
        print(f"\nAMBIGUOUS: Not enough samples ({n_ambig} < 10)")


if __name__ == "__main__":
    main()
