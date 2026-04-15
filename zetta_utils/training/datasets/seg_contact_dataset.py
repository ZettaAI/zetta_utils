from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import attrs
import numpy as np
import torch

from zetta_utils import builder, log

logger = log.get_logger("zetta_utils")
from zetta_utils.layer.volumetric.seg_contact import (
    SegContact,
    VolumetricSegContactLayer,
)
from zetta_utils.layer.volumetric.seg_contact.tensor_utils import (
    contact_faces_to_tensor,
    pointcloud_to_labeled_tensor,
)

from .sample_indexers import SampleIndexer


def _pad_or_truncate(tensor: torch.Tensor, target_size: int) -> torch.Tensor:
    """Pad with zeros or truncate tensor to target size along first dimension."""
    current_size = tensor.shape[0]
    if current_size == target_size:
        return tensor
    if current_size > target_size:
        return tensor[:target_size]
    padding = torch.zeros(target_size - current_size, *tensor.shape[1:], dtype=tensor.dtype)
    return torch.cat([tensor, padding], dim=0)


def _broadcast_to_list(value: float | str | list, n_channels: int) -> list:
    """Broadcast a single value to a list of length n_channels, or return list as-is."""
    if isinstance(value, list):
        return value
    return [value] * n_channels


def _apply_channel_mask(
    pointcloud: torch.Tensor,
    mask_probs: list[float] | float,
    mask_modes: list[str] | str,
    mask_global_probs: list[float] | float,
    mask_value: float,
) -> torch.Tensor:
    """Apply random channel masking to pointcloud feature channels (not xyz).

    Only operates on channels starting from index 3 (i.e., skips xyz).
    The mask config indices map to channels 3, 4, 5, etc.

    Args:
        pointcloud: Tensor of shape [n_points, n_channels]
        mask_probs: Masking probability for feature channels (index 0 = channel 3, etc.)
            Single value broadcasts to all feature channels.
        mask_modes: Mode per feature channel - "global", "local", or "random" (broadcasts)
        mask_global_probs: Probability of global mode when mode="random" (broadcasts)
        mask_value: Value to use for masked entries

    Returns:
        Masked pointcloud tensor
    """
    n_points, n_channels = pointcloud.shape
    n_feature_channels = n_channels - 3  # Skip xyz

    if n_feature_channels <= 0:
        return pointcloud

    # Broadcast single values to all feature channels
    probs = _broadcast_to_list(mask_probs, n_feature_channels)
    modes = _broadcast_to_list(mask_modes, n_feature_channels)
    global_probs = _broadcast_to_list(mask_global_probs, n_feature_channels)

    result = pointcloud.clone()

    for feat_idx in range(n_feature_channels):
        ch_idx = feat_idx + 3  # Actual channel index (skip xyz)
        prob = probs[feat_idx] if feat_idx < len(probs) else 0.0
        if prob <= 0:
            continue

        # Check if this channel should be masked
        if torch.rand(1).item() >= prob:
            continue

        # Determine mode for this channel
        mode = modes[feat_idx] if feat_idx < len(modes) else "global"
        if mode == "random":
            global_prob = global_probs[feat_idx] if feat_idx < len(global_probs) else 0.5
            mode = "global" if torch.rand(1).item() < global_prob else "local"

        if mode == "global":
            # Mask entire channel
            result[:, ch_idx] = mask_value
        else:  # local
            # Random mask per point (using the same prob)
            point_mask = torch.rand(n_points) < prob
            result[point_mask, ch_idx] = mask_value

    return result


@builder.register("SegContactDataset")
@attrs.frozen
class SegContactDataset(torch.utils.data.Dataset):
    """PyTorch dataset for seg_contact data - returns chunk-aligned batches.

    Uses layer + indexer architecture:
    - Indexer returns VolumetricIndex (bounding box) for each chunk
    - Layer reads contacts via read_with_procs (enabling augmentations/normalization)

    Outputs concatenated pointclouds from all available configs in sorted order.
    Each pointcloud has xyz + 4th channel (+ optional 5th channel):
    - For local pointclouds: 4th channel is segment_label (-1 for seg_a, +1 for seg_b)
    - For contact_faces: 4th channel is contact_label (default 0.0) or affinity if None
    - Optional 5th channel: affinity values (0 for segment points, affinity for CF points)

    :param layer: VolumetricSegContactLayer to read contacts from.
    :param sample_indexer: Indexer that returns VolumetricIndex per chunk.
    :param merge_decision_authority: Authority name for merge decisions.
    :param max_contact_faces: Fixed size for contact_faces tensor (pad/truncate).
    :param include_contact_faces_in_pointcloud: If True, append contact_faces as
        additional points in the pointcloud.
    :param contact_label: Label for contact faces in pointcloud. If float (default 0.0),
        use this value as the 4th channel. If None, use affinity as 4th channel.
    :param affinity_channel_mode: Mode for optional 5th affinity channel:
        - None: No 5th channel (backward compatible, 4-channel output)
        - "per_point": 5th channel has per-point affinity for CF points, 0 for segments
        - "mean": 5th channel has mean affinity broadcast to all CF points, 0 for segments
    :param mask_channel_probs: Masking probability for feature channels (skips xyz).
        Index 0 = channel 3 (segment_label), index 1 = channel 4 (affinity), etc. Can be:
        - None: No masking (default)
        - float: Same probability for all feature channels
        - list[float]: Per-feature-channel probabilities
        E.g., [0.5, 0.3] masks segment_label with 50%, affinity with 30%.
    :param mask_mode: Masking mode for feature channels (broadcasts if single value):
        - "global": Mask entire channel for all points
        - "local": Randomly mask each point independently
        - "random": Randomly choose between global and local per sample
    :param mask_mode_global_prob: When mask_mode="random", probability of using global mode.
        Broadcasts if single value, or per-feature-channel list.
    :param mask_value: Value to use for masked entries (default 0.0).
    :param affinity_noise_std: Std of Gaussian noise to add to affinity values in the 5th
        channel (requires affinity_channel_mode to be set). None disables noise.
    :param affinity_noise_prob: Probability of applying affinity noise per sample.
    """

    layer: VolumetricSegContactLayer
    sample_indexer: SampleIndexer
    merge_decision_authority: str | None = None
    max_contact_faces: int = 2048
    include_contact_faces_in_pointcloud: bool = False
    contact_label: float | None = 0.0
    affinity_channel_mode: str | None = None
    mask_channel_probs: list[float] | float | None = None
    mask_mode: list[str] | str = "global"
    mask_mode_global_prob: list[float] | float = 0.5
    mask_value: float = 0.0
    affinity_noise_std: float | None = None
    affinity_noise_prob: float = 0.5
    min_mean_affinity: float | None = None
    max_mean_affinity: float | None = None

    def __attrs_pre_init__(self):
        super().__init__()

    def __attrs_post_init__(self):
        if self.affinity_channel_mode is not None:
            assert self.include_contact_faces_in_pointcloud, (
                "affinity_channel_mode requires include_contact_faces_in_pointcloud=True. "
                "Otherwise all points would have 0 in the affinity channel."
            )

    def __len__(self) -> int:
        return len(self.sample_indexer)

    def __getitem__(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        self, idx: int
    ) -> dict[str, Any]:
        # Get bounding box from indexer, then read contacts via layer
        vol_idx = self.sample_indexer(idx)
        contacts: Sequence[SegContact] = self.layer[vol_idx]

        if not contacts:
            return {
                "contact_id": torch.tensor([], dtype=torch.int64),
                "seg_a": torch.tensor([], dtype=torch.int64),
                "seg_b": torch.tensor([], dtype=torch.int64),
                "com": torch.zeros((0, 3), dtype=torch.float32),
                "contact_faces": torch.zeros((0, self.max_contact_faces, 4), dtype=torch.float32),
                "n_contact_faces": torch.tensor([], dtype=torch.int32),
                "mean_affinity": torch.tensor([], dtype=torch.float32),
            }

        # Get sorted config keys from first contact that has pointclouds
        config_keys: list[tuple[int, int]] = []
        for contact in contacts:
            if contact.local_pointclouds:
                config_keys = sorted(contact.local_pointclouds.keys())
                break

        has_affinity = self.affinity_channel_mode is not None
        n_channels = 5 if has_affinity else 4
        n_contacts = len(contacts)
        max_cf = self.max_contact_faces

        # Pre-allocate numpy arrays (over-allocate for n_contacts, trim later)
        contact_ids = np.empty(n_contacts, dtype=np.int64)
        seg_as = np.empty(n_contacts, dtype=np.int64)
        seg_bs = np.empty(n_contacts, dtype=np.int64)
        coms = np.empty((n_contacts, 3), dtype=np.float32)
        cf_all = np.zeros((n_contacts, max_cf, 4), dtype=np.float32)
        cf_orig_all = np.zeros((n_contacts, max_cf, 4), dtype=np.float32)
        n_contact_faces = np.zeros(n_contacts, dtype=np.int32)
        mean_affinities = np.zeros(n_contacts, dtype=np.float32)
        targets = np.empty(n_contacts, dtype=np.float32)
        info_path_list: list[str] = []

        # Pointcloud parts collected per contact in numpy, then assembled
        pointclouds_np: list[np.ndarray] = []
        n_valid = 0
        n_targets = 0
        has_cf_orig_warning = False
        info_path = f"{self.layer.backend.path}/info"

        for contact in contacts:
            skip_contact = False

            # Compute mean affinity and filter early
            cf_for_aff = contact.contact_faces_original_nm
            if cf_for_aff is None:
                cf_for_aff = contact.contact_faces
            mean_aff = float(cf_for_aff[:, 3].mean()) if cf_for_aff.shape[0] > 0 else 0.0
            if self.min_mean_affinity is not None and mean_aff < self.min_mean_affinity:
                skip_contact = True
            if self.max_mean_affinity is not None and mean_aff > self.max_mean_affinity:
                skip_contact = True
            if skip_contact:
                continue

            # Skip contacts without pointclouds (e.g. missing mesh data)
            if contact.local_pointclouds is None or not config_keys:
                logger.debug(
                    f"Skipping contact {contact.id} (seg_a={contact.seg_a}, seg_b={contact.seg_b}): "
                    f"no local_pointclouds (has {contact.contact_faces.shape[0]} contact faces)"
                )
                continue
            pc_parts_np: list[np.ndarray] = []
            for config_key in config_keys:
                pc_data = None
                if contact.local_pointclouds is not None:
                    pc_data = contact.local_pointclouds.get(config_key)

                if pc_data is not None:
                    seg_a_pts = pc_data.get(contact.seg_a)
                    seg_b_pts = pc_data.get(contact.seg_b)
                else:
                    seg_a_pts = None
                    seg_b_pts = None

                if seg_a_pts is None or seg_b_pts is None:
                    skip_contact = True
                    break

                # Build labeled pointcloud in numpy: [N, 4 or 5]
                n_a, n_b = seg_a_pts.shape[0], seg_b_pts.shape[0]
                seg_a_labeled = np.empty((n_a, n_channels), dtype=np.float32)
                seg_a_labeled[:, :3] = seg_a_pts
                seg_a_labeled[:, 3] = -1.0
                seg_b_labeled = np.empty((n_b, n_channels), dtype=np.float32)
                seg_b_labeled[:, :3] = seg_b_pts
                seg_b_labeled[:, 3] = 1.0
                if has_affinity:
                    seg_a_labeled[:, 4] = 0.0
                    seg_b_labeled[:, 4] = 0.0
                pc_parts_np.append(seg_a_labeled)
                pc_parts_np.append(seg_b_labeled)

            if skip_contact:
                continue

            i = n_valid

            # Scalar fields
            contact_ids[i] = contact.id
            seg_as[i] = contact.seg_a
            seg_bs[i] = contact.seg_b
            coms[i] = contact.com

            # Contact faces - pad or truncate into pre-allocated array
            cf = contact.contact_faces
            cf_n = min(cf.shape[0], max_cf)
            cf_all[i, :cf_n] = cf[:cf_n]
            n_contact_faces[i] = cf_n
            mean_affinities[i] = mean_aff

            if contact.contact_faces_original_nm is not None:
                cf_orig = contact.contact_faces_original_nm
                cf_orig_n = min(cf_orig.shape[0], max_cf)
                cf_orig_all[i, :cf_orig_n] = cf_orig[:cf_orig_n]
            elif not has_cf_orig_warning:
                warnings.warn(
                    f"contact_faces_original_nm is None for contact {contact.id}. "
                    "NG links will show normalized coordinates (around origin). "
                    "Ensure normalize_pointclouds runs before other augmentations.",
                    stacklevel=2,
                )
                has_cf_orig_warning = True
                cf_orig_all[i, :cf_n] = cf[:cf_n]

            # Contact faces as points in pointcloud (cf is unpadded, all rows valid)
            if self.include_contact_faces_in_pointcloud and cf.shape[0] > 0:
                n_cf = cf.shape[0]
                cf_labeled = np.empty((n_cf, n_channels), dtype=np.float32)
                cf_labeled[:, :3] = cf[:, :3]
                if self.contact_label is not None:
                    cf_labeled[:, 3] = self.contact_label
                else:
                    cf_labeled[:, 3] = cf[:, 3]
                if has_affinity:
                    if self.affinity_channel_mode == "per_point":
                        cf_labeled[:, 4] = cf[:, 3]
                    elif self.affinity_channel_mode == "mean":
                        cf_labeled[:, 4] = cf[:, 3].mean()
                pc_parts_np.append(cf_labeled)

            if pc_parts_np:
                contact_pc = np.concatenate(pc_parts_np, axis=0)
                # Pointcloud n_points must be consistent across all contacts in a chunk
                # (required by np.stack and downstream RebatchingDataLoader torch.cat)
                if pointclouds_np:
                    expected_n = pointclouds_np[0].shape[0]
                    assert contact_pc.shape[0] == expected_n, (
                        f"Pointcloud size mismatch: contact {contact.id} (seg_a={contact.seg_a}, "
                        f"seg_b={contact.seg_b}) has {contact_pc.shape[0]} points, expected {expected_n}. "
                        f"Parts: {[p.shape[0] for p in pc_parts_np]}. "
                        f"Check that resample_combined_pointcloud is in read_procs."
                    )
                pointclouds_np.append(contact_pc)

            # Target
            if self.merge_decision_authority is not None and contact.merge_decisions is not None:
                should_merge = contact.merge_decisions.get(self.merge_decision_authority)
                if should_merge is not None:
                    targets[n_targets] = 1.0 if should_merge else 0.0
                    n_targets += 1

            info_path_list.append(info_path)
            n_valid += 1

        # Trim to actual valid count and convert to torch once
        result: dict[str, Any] = {
            "contact_id": torch.from_numpy(contact_ids[:n_valid].copy()),
            "seg_a": torch.from_numpy(seg_as[:n_valid].copy()),
            "seg_b": torch.from_numpy(seg_bs[:n_valid].copy()),
            "com": torch.from_numpy(coms[:n_valid].copy()),
            "contact_faces": torch.from_numpy(cf_all[:n_valid].copy()),
            "contact_faces_original_nm": torch.from_numpy(cf_orig_all[:n_valid].copy()),
            "n_contact_faces": torch.from_numpy(n_contact_faces[:n_valid].copy()),
            "mean_affinity": torch.from_numpy(mean_affinities[:n_valid].copy()),
            "info_path": info_path_list,
        }

        if not pointclouds_np:
            logger.debug(
                f"Chunk has {n_valid} contacts but none with valid pointclouds — "
                f"returning empty result so RebatchingDataLoader skips this chunk"
            )
            return {}

        if pointclouds_np:
            # Stack pointclouds — apply torch-level augmentations if needed
            pointclouds_stacked = torch.from_numpy(np.stack(pointclouds_np))

            if self.affinity_noise_std is not None and self.affinity_noise_std > 0:
                if pointclouds_stacked.shape[2] > 4:
                    seg_labels = pointclouds_stacked[:, :, 3]
                    is_cf = (seg_labels != -1.0) & (seg_labels != 1.0)
                    for pi in range(pointclouds_stacked.shape[0]):
                        if torch.rand(1).item() < self.affinity_noise_prob:
                            noise = (
                                torch.randn(pointclouds_stacked.shape[1]) * self.affinity_noise_std
                            )
                            noise = noise * is_cf[pi].to(noise.dtype)
                            pointclouds_stacked[pi, :, 4] += noise

            if self.mask_channel_probs is not None:
                for pi in range(pointclouds_stacked.shape[0]):
                    pointclouds_stacked[pi] = _apply_channel_mask(
                        pointclouds_stacked[pi],
                        self.mask_channel_probs,
                        self.mask_mode,
                        self.mask_mode_global_prob,
                        self.mask_value,
                    )

            result["pointcloud"] = pointclouds_stacked

        if n_targets > 0:
            result["merge"] = torch.from_numpy(targets[:n_targets, np.newaxis].copy())
        elif pointclouds_np:
            logger.debug(
                f"Chunk has {len(pointclouds_np)} contacts with pointclouds but no targets — "
                f"returning empty result so RebatchingDataLoader skips this chunk"
            )
            return {}

        return result
