#!/usr/bin/env python3
"""Compute aggregate metrics for contact merge predictions across all chunks.

Reads written merge_probabilities and compares to ground truth merge_decisions.
Computes AUC-PR, AUC-ROC, precision, recall for ALL and AMBIGUOUS (MA 0.15-0.35) samples.

Usage:
    python compute_merge_metrics.py \
        --path gs://martin_exp/contact_merge_inference_test/fafb_v14_cutout_x2_contacts_x1 \
        --authority model_v6.0_r2k_p2k_pn4k_cf_aff_bs64_minbs16_lr1e-3_test \
        --gt-authority ground_truth \
        --bbox 29664,9364,1240,34808,12460,1304 \
        --chunk-size 256,256,64
"""

import argparse
import struct
import io

import numpy as np
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


def get_chunk_keys(start: list[int], end: list[int], chunk_size: list[int]) -> list[str]:
    """Generate chunk keys for all chunks in bbox."""
    keys = []
    for x in range(start[0], end[0], chunk_size[0]):
        for y in range(start[1], end[1], chunk_size[1]):
            for z in range(start[2], end[2], chunk_size[2]):
                x_end = min(x + chunk_size[0], end[0])
                y_end = min(y + chunk_size[1], end[1])
                z_end = min(z + chunk_size[2], end[2])
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

            # Compute mean affinity
            if faces:
                mean_affinity = np.mean([f[3] for f in faces])
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


def compute_metrics(
    targets: np.ndarray,
    probs: np.ndarray,
    mean_affinities: np.ndarray,
    subset_name: str,
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
    parser.add_argument("--path", required=True, help="Seg_contact path with predictions")
    parser.add_argument("--authority", required=True, help="Merge probability authority name")
    parser.add_argument("--gt-authority", default="ground_truth", help="Ground truth authority name")
    parser.add_argument("--bbox", required=True, help="Bounding box: x0,y0,z0,x1,y1,z1")
    parser.add_argument("--resolution", default="16,16,40", help="Resolution: x,y,z")
    parser.add_argument("--chunk-size", default="256,256,64", help="Chunk size: x,y,z")
    parser.add_argument("--min-mean-affinity", type=float, default=0.1, help="Min mean affinity filter")
    args = parser.parse_args()

    start, end = parse_bbox(args.bbox)
    chunk_size = parse_vec(args.chunk_size)
    chunk_keys = get_chunk_keys(start, end, chunk_size)

    print(f"Path: {args.path}")
    print(f"Authority: {args.authority}")
    print(f"GT Authority: {args.gt_authority}")
    print(f"BBox: {start} -> {end}")
    print(f"Chunk size: {chunk_size}")
    print(f"Total chunks: {len(chunk_keys)}")
    print(f"Min mean affinity: {args.min_mean_affinity}")

    # Parse bucket and path
    bucket_name = args.path.replace("gs://", "").split("/")[0]
    base_path = "/".join(args.path.replace("gs://", "").split("/")[1:])

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Collect data from all chunks
    all_targets = []
    all_probs = []
    all_mean_affinities = []

    n_chunks_with_data = 0
    n_contacts_total = 0
    n_with_gt = 0
    n_with_pred = 0
    n_matched = 0

    for chunk_key in tqdm(chunk_keys, desc="Reading chunks"):
        # Read contacts (for mean_affinity)
        contacts = read_contacts_chunk(
            bucket, f"{base_path}/contacts/{chunk_key}"
        )

        # Read ground truth
        gt_decisions = read_merge_decisions_chunk(
            bucket, f"{base_path}/merge_decisions/{args.gt_authority}/{chunk_key}"
        )

        # Read predictions
        predictions = read_merge_probabilities_chunk(
            bucket, f"{base_path}/merge_probabilities/{args.authority}/{chunk_key}"
        )

        if not contacts:
            continue

        n_chunks_with_data += 1
        n_contacts_total += len(contacts)
        n_with_gt += len(gt_decisions)
        n_with_pred += len(predictions)

        # Match contacts with both GT and predictions
        for contact_id, contact in contacts.items():
            if contact_id not in gt_decisions:
                continue
            if contact_id not in predictions:
                continue

            mean_aff = contact["mean_affinity"]
            if mean_aff < args.min_mean_affinity:
                continue

            n_matched += 1
            all_targets.append(1.0 if gt_decisions[contact_id] else 0.0)
            all_probs.append(predictions[contact_id])
            all_mean_affinities.append(mean_aff)

    print(f"\nData collection:")
    print(f"  Chunks with data: {n_chunks_with_data}/{len(chunk_keys)}")
    print(f"  Total contacts: {n_contacts_total}")
    print(f"  With ground truth: {n_with_gt}")
    print(f"  With predictions: {n_with_pred}")
    print(f"  Matched (after affinity filter): {n_matched}")

    if n_matched < 10:
        print("\nERROR: Not enough matched samples for metrics computation")
        return

    targets = np.array(all_targets)
    probs = np.array(all_probs)
    mean_affinities = np.array(all_mean_affinities)

    # Compute metrics for ALL samples
    compute_metrics(targets, probs, mean_affinities, "ALL")

    # Compute metrics for AMBIGUOUS samples (MA 0.15-0.35)
    ambig_mask = (mean_affinities >= 0.15) & (mean_affinities <= 0.35)
    n_ambig = ambig_mask.sum()
    if n_ambig >= 10:
        compute_metrics(
            targets[ambig_mask],
            probs[ambig_mask],
            mean_affinities[ambig_mask],
            f"AMBIGUOUS (MA 0.15-0.35, n={n_ambig})",
        )
    else:
        print(f"\nAMBIGUOUS: Not enough samples ({n_ambig} < 10)")


if __name__ == "__main__":
    main()
