"""
Cutie VOS mask propagation for EM segmentation data.

Propagates a segmentation mask from the first frame through consecutive
image slices using the Cutie video object segmentation model.

Weights are stored in gs://zetta_ws/models/cutie/ (Zetta AI access only)
and auto-downloaded on first use.

Requires: pip install cutie (from https://github.com/hkchengrex/Cutie)
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import zoom

WEIGHTS_DIR = Path.home() / ".cache" / "cutie" / "weights"
GCS_WEIGHTS_PREFIX = "gs://zetta_ws/models/cutie"
WEIGHT_FILES = ["cutie-base-mega.pth", "coco_lvis_h18_itermask.pth"]


def _ensure_weights() -> Path:
    """Download Cutie weights from GCS if not present locally."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for fname in WEIGHT_FILES:
        local_path = WEIGHTS_DIR / fname
        if not local_path.exists():
            gcs_path = f"{GCS_WEIGHTS_PREFIX}/{fname}"
            print(f"Downloading {fname} from {gcs_path}...")
            subprocess.run(
                ["gsutil", "cp", gcs_path, str(local_path)],
                check=True,
            )
            print(f"  Saved to {local_path}")
    return WEIGHTS_DIR


def _load_cutie(device: str = "cpu"):
    """Load the Cutie model with weights from local cache."""
    import cutie.config as cutie_config
    from cutie.inference.utils.args_utils import get_dataset_cfg
    from cutie.model.cutie import CUTIE
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict

    weight_dir = _ensure_weights()
    config_dir = str(Path(cutie_config.__path__[0]))
    with initialize_config_dir(version_base="1.3.2", config_dir=config_dir):
        cfg = compose(config_name="eval_config")

    with open_dict(cfg):
        cfg["weights"] = str(weight_dir / "cutie-base-mega.pth")
    get_dataset_cfg(cfg)

    model = CUTIE(cfg).to(device).eval()
    model_weights = torch.load(cfg.weights, map_location=device, weights_only=True)
    model.load_weights(model_weights)

    return model


def propagate_mask(
    images: list[np.ndarray],
    initial_mask: np.ndarray,
    object_ids: list[int] | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> list[np.ndarray]:
    """Propagate a segmentation mask through consecutive frames using Cutie VOS.

    Args:
        images: List of grayscale 2D images, each (H, W), uint8.
        initial_mask: Binary or labeled mask for the first frame, (H, W).
            Non-zero pixels are treated as the object. Must match image dimensions.
        object_ids: List of object IDs present in initial_mask. If None,
            defaults to [1] and treats any non-zero pixel as object 1.
        device: "cuda" or "cpu".

    Returns:
        List of predicted binary masks (bool, H, W) for each frame,
        including the first frame (which uses the provided mask directly).
    """
    from cutie.inference.inference_core import InferenceCore

    assert len(images) > 0, "Need at least one image"
    assert (
        images[0].shape == initial_mask.shape
    ), f"Image shape {images[0].shape} != mask shape {initial_mask.shape}"

    if object_ids is None:
        object_ids = [1]

    cutie = _load_cutie(device)
    processor = InferenceCore(cutie, cfg=cutie.cfg)

    predicted_masks: list[np.ndarray] = []

    with torch.inference_mode():
        for i, img in enumerate(images):
            # Grayscale -> 3-channel, scale to [0,1] float32. No ImageNet norm.
            frame = np.stack([img] * 3, axis=0).astype(np.float32) / 255.0
            frame_tensor = torch.from_numpy(frame).to(device)

            if i == 0:
                mask_tensor = torch.from_numpy(initial_mask.astype(np.int64)).to(device)
                processor.step(frame_tensor, mask_tensor, objects=object_ids)
                predicted_masks.append(initial_mask > 0)
            else:
                output_prob = processor.step(frame_tensor)
                pred = (output_prob.argmax(0) > 0).cpu().numpy()
                predicted_masks.append(pred)

    return predicted_masks


def evaluate_masks(
    predicted_masks: list[np.ndarray],
    ground_truth_masks: list[np.ndarray],
) -> list[dict[str, float]]:
    """Compute IoU and Dice between predicted and ground truth masks.

    Args:
        predicted_masks: List of binary masks (bool or 0/1), each (H, W).
        ground_truth_masks: List of binary masks, same length and shapes.

    Returns:
        List of dicts with keys "iou" and "dice" for each frame pair.
    """
    assert len(predicted_masks) == len(
        ground_truth_masks
    ), f"Length mismatch: {len(predicted_masks)} vs {len(ground_truth_masks)}"

    results = []
    for pred, gt in zip(predicted_masks, ground_truth_masks):
        p = pred.astype(bool)
        g = gt.astype(bool)
        intersection = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()

        iou = float(intersection / union) if union > 0 else 1.0
        dice = float(2 * intersection / (p.sum() + g.sum())) if (p.sum() + g.sum()) > 0 else 1.0

        results.append({"iou": iou, "dice": dice})

    return results


def save_overlay_images(
    images: list[np.ndarray],
    predicted_masks: list[np.ndarray],
    output_dir: str | Path,
    ground_truth_masks: list[np.ndarray] | None = None,
    alpha: float = 0.4,
) -> None:
    """Save images with mask overlays as PNGs.

    Args:
        images: List of grayscale (H, W) uint8 images.
        predicted_masks: Predicted binary masks.
        output_dir: Directory to save overlay images.
        ground_truth_masks: If provided, shown in green; predicted in red; overlap in yellow.
        alpha: Overlay transparency (0=invisible, 1=opaque).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, (img, pred) in enumerate(zip(images, predicted_masks)):
        rgb = np.stack([img, img, img], axis=-1).astype(np.float32)
        pred_bool = pred.astype(bool)

        overlay = np.zeros_like(rgb)
        overlay[pred_bool] = [255, 0, 0]  # red for predicted

        if ground_truth_masks is not None:
            gt_bool = ground_truth_masks[i].astype(bool)
            overlay[gt_bool] = [0, 255, 0]  # green for GT
            overlap = pred_bool & gt_bool
            overlay[overlap] = [255, 255, 0]  # yellow for overlap

        blended = ((1 - alpha) * rgb + alpha * overlay).clip(0, 255).astype(np.uint8)
        Image.fromarray(blended).save(output_dir / f"slice_{i:04d}.png")

    print(f"Saved {len(images)} overlay images to {output_dir}")


if __name__ == "__main__":
    DATA_DIR = "/home/sergiy/code/zetta-terminal"
    OBJECT_ID = 73466275598936721

    print("Loading data...")
    raw = np.load(f"{DATA_DIR}/ant_cutout_50z.npy")  # (526, 430, 50, 1) uint8
    seg = np.load(f"{DATA_DIR}/ant_seg_50z.npy")  # (263, 215, 50, 1) uint64

    n_slices = raw.shape[2]
    print(f"Raw shape: {raw.shape}, Seg shape: {seg.shape}, Slices: {n_slices}")

    # Extract images: .T on [:,:,i,0] to get (H, W)
    images = [raw[:, :, i, 0].T for i in range(n_slices)]
    H, W = images[0].shape
    print(f"Image size: {H} x {W}")

    # Extract and resize ground truth masks (seg is 2x coarser)
    gt_masks = []
    for i in range(n_slices):
        seg_slice = seg[:, :, i, 0].T
        binary = (seg_slice == OBJECT_ID).astype(np.float64)
        scale_h = H / binary.shape[0]
        scale_w = W / binary.shape[1]
        resized = zoom(binary, (scale_h, scale_w), order=0)
        gt_masks.append(resized > 0.5)

    initial_mask = gt_masks[0].astype(np.int64)
    print(f"Object pixels in first frame: {initial_mask.sum()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Propagating mask with Cutie...")
    predicted = propagate_mask(images, initial_mask, object_ids=[1], device=device)

    print("Evaluating...")
    metrics = evaluate_masks(predicted, gt_masks)

    print(f"\n{'Slice':>6}  {'IoU':>7}  {'Dice':>7}")
    print("-" * 25)
    for i, m in enumerate(metrics):
        if i % 10 == 0 or i == n_slices - 1:
            print(f"{i:>6}  {m['iou']:>7.4f}  {m['dice']:>7.4f}")

    avg_iou = np.mean([m["iou"] for m in metrics])
    avg_dice = np.mean([m["dice"] for m in metrics])
    print(f"\nAverage IoU:  {avg_iou:.4f}")
    print(f"Average Dice: {avg_dice:.4f}")

    # Save overlay images
    output_dir = Path(DATA_DIR) / "cutie_overlays"
    print(f"\nSaving overlay images to {output_dir}...")
    save_overlay_images(images, predicted, output_dir, ground_truth_masks=gt_masks)
