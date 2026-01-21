from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import attrs
import torch

from zetta_utils import builder
from zetta_utils.layer.volumetric.seg_contact import (
    SegContact,
    VolumetricSegContactLayer,
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
            }

        # Build batched tensors
        contact_ids = []
        seg_as = []
        seg_bs = []
        coms = []
        contact_faces_list = []
        contact_faces_original_nm_list = []
        pointclouds_list = []  # List of [n_points, 4 or 5] tensors per contact
        targets = []

        # Get sorted config keys from first contact that has pointclouds
        config_keys: list[tuple[int, int]] = []
        for contact in contacts:
            if contact.local_pointclouds:
                config_keys = sorted(contact.local_pointclouds.keys())
                break

        for contact in contacts:
            # Build concatenated pointcloud for this contact
            contact_pointcloud_parts = []

            skip_contact = False
            for config_key in config_keys:
                # config_key is (radius_nm, n_points)
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
                    # Missing pointcloud config - skip this contact
                    skip_contact = True
                    break

                # Add segment labels: -1 for seg_a, +1 for seg_b
                seg_a_tensor = torch.tensor(seg_a_pts, dtype=torch.float32)
                seg_b_tensor = torch.tensor(seg_b_pts, dtype=torch.float32)

                n_a, n_b = seg_a_tensor.shape[0], seg_b_tensor.shape[0]

                labels_a = torch.full((n_a, 1), -1.0, dtype=torch.float32)
                labels_b = torch.full((n_b, 1), 1.0, dtype=torch.float32)

                seg_a_with_label = torch.cat([seg_a_tensor, labels_a], dim=-1)
                seg_b_with_label = torch.cat([seg_b_tensor, labels_b], dim=-1)

                # Add 5th channel (affinity) if requested - 0 for segment points
                if self.affinity_channel_mode is not None:
                    zeros_a = torch.zeros((n_a, 1), dtype=torch.float32)
                    zeros_b = torch.zeros((n_b, 1), dtype=torch.float32)
                    seg_a_with_label = torch.cat([seg_a_with_label, zeros_a], dim=-1)
                    seg_b_with_label = torch.cat([seg_b_with_label, zeros_b], dim=-1)

                contact_pointcloud_parts.append(seg_a_with_label)
                contact_pointcloud_parts.append(seg_b_with_label)

            if skip_contact:
                continue

            # Contact has all required pointclouds - add to batch
            contact_ids.append(contact.id)
            seg_as.append(contact.seg_a)
            seg_bs.append(contact.seg_b)
            coms.append(list(contact.com))

            # Contact faces
            contact_faces_tensor = _pad_or_truncate(
                torch.tensor(contact.contact_faces, dtype=torch.float32),
                self.max_contact_faces,
            )
            contact_faces_list.append(contact_faces_tensor)

            # Original contact faces in nm (for visualization)
            if contact.contact_faces_original_nm is not None:
                contact_faces_original_nm_tensor = _pad_or_truncate(
                    torch.tensor(contact.contact_faces_original_nm, dtype=torch.float32),
                    self.max_contact_faces,
                )
            else:
                # Fallback to normalized if original not available
                warnings.warn(
                    f"contact_faces_original_nm is None for contact {contact.id}. "
                    "NG links will show normalized coordinates (around origin). "
                    "Ensure normalize_pointclouds runs before other augmentations.",
                    stacklevel=2,
                )
                contact_faces_original_nm_tensor = contact_faces_tensor
            contact_faces_original_nm_list.append(contact_faces_original_nm_tensor)

            # Optionally include contact faces as additional points
            if self.include_contact_faces_in_pointcloud:
                # Filter out zero-padded contact faces
                cf_tensor = torch.tensor(contact.contact_faces, dtype=torch.float32)
                valid_mask = torch.any(cf_tensor[:, :3] != 0, dim=1)
                valid_cf = cf_tensor[valid_mask]

                if valid_cf.shape[0] > 0:
                    # Get per-point affinities before potentially overwriting
                    per_point_affinities = valid_cf[:, 3].clone()

                    if self.contact_label is not None:
                        # Replace affinity with contact_label
                        cf_with_label = valid_cf.clone()
                        cf_with_label[:, 3] = self.contact_label
                    else:
                        # Keep affinity as 4th channel
                        cf_with_label = valid_cf.clone()

                    # Add 5th channel (affinity) if requested
                    if self.affinity_channel_mode is not None:
                        n_cf = cf_with_label.shape[0]
                        if self.affinity_channel_mode == "per_point":
                            # Per-point affinity values
                            aff_channel = per_point_affinities.unsqueeze(-1)
                        elif self.affinity_channel_mode == "mean":
                            # Mean affinity broadcast to all CF points
                            mean_aff = per_point_affinities.mean()
                            aff_channel = torch.full((n_cf, 1), mean_aff, dtype=torch.float32)
                        else:
                            raise ValueError(
                                f"Unknown affinity_channel_mode: {self.affinity_channel_mode}. "
                                "Must be 'per_point', 'mean', or None."
                            )
                        cf_with_label = torch.cat([cf_with_label, aff_channel], dim=-1)

                    contact_pointcloud_parts.append(cf_with_label)

            if contact_pointcloud_parts:
                contact_pointcloud = torch.cat(contact_pointcloud_parts, dim=0)

                # Add Gaussian noise to affinity channel (5th channel, index 4) if configured
                if self.affinity_noise_std is not None and self.affinity_noise_std > 0:
                    if contact_pointcloud.shape[1] > 4:  # Has 5th channel
                        if torch.rand(1).item() < self.affinity_noise_prob:
                            noise = torch.randn(contact_pointcloud.shape[0]) * self.affinity_noise_std
                            contact_pointcloud[:, 4] = contact_pointcloud[:, 4] + noise

                # Apply channel masking if configured
                if self.mask_channel_probs is not None:
                    contact_pointcloud = _apply_channel_mask(
                        contact_pointcloud,
                        self.mask_channel_probs,
                        self.mask_mode,
                        self.mask_mode_global_prob,
                        self.mask_value,
                    )

                pointclouds_list.append(contact_pointcloud)

            # Target
            if self.merge_decision_authority is not None and contact.merge_decisions is not None:
                should_merge = contact.merge_decisions.get(self.merge_decision_authority)
                if should_merge is not None:
                    targets.append(1.0 if should_merge else 0.0)

        info_path = f"{self.layer.backend.path}/info"
        result: dict[str, Any] = {
            "contact_id": torch.tensor(contact_ids, dtype=torch.int64),
            "seg_a": torch.tensor(seg_as, dtype=torch.int64),
            "seg_b": torch.tensor(seg_bs, dtype=torch.int64),
            "com": torch.tensor(coms, dtype=torch.float32),
            "contact_faces": torch.stack(contact_faces_list),
            "contact_faces_original_nm": torch.stack(contact_faces_original_nm_list),
            "info_path": [info_path] * len(contact_ids),
        }

        if pointclouds_list:
            result["pointcloud"] = torch.stack(pointclouds_list)  # [B, total_points, 4 or 5]
        if targets:
            result["target"] = torch.tensor(targets, dtype=torch.float32)

        return result
