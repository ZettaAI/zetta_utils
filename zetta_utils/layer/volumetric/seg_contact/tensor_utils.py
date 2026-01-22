"""Utility functions for converting seg_contact data to tensors.

This module provides shared conversion utilities used by both training (SegContactDataset)
and inference (ContactMergeInferencer) to ensure consistent data representation.

IMPORTANT: Segment labels use -1 for seg_a and +1 for seg_b. This convention must be
consistent between training and inference.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from .contact import SegContact

# Segment label constants - MUST be consistent between training and inference
SEG_A_LABEL = -1.0
SEG_B_LABEL = 1.0


def pointcloud_to_labeled_tensor(
    seg_a_pts: np.ndarray,
    seg_b_pts: np.ndarray,
    affinity_channel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert segment pointclouds to labeled tensors.

    This is the core conversion that MUST be shared between training and inference
    to ensure consistent segment labeling.

    Args:
        seg_a_pts: Points for segment A, shape [N_a, 3].
        seg_b_pts: Points for segment B, shape [N_b, 3].
        affinity_channel: If True, add a 5th channel (0 for segment points).

    Returns:
        Tuple of (seg_a_tensor, seg_b_tensor), each shape [N, 4] or [N, 5] where
        the 4th channel is the segment label (-1 for A, +1 for B), and optional
        5th channel is affinity (0 for segment points).
    """
    seg_a_tensor = torch.tensor(seg_a_pts, dtype=torch.float32)
    seg_b_tensor = torch.tensor(seg_b_pts, dtype=torch.float32)

    n_a, n_b = seg_a_tensor.shape[0], seg_b_tensor.shape[0]

    # Add segment labels: -1 for seg_a, +1 for seg_b
    labels_a = torch.full((n_a, 1), SEG_A_LABEL, dtype=torch.float32)
    labels_b = torch.full((n_b, 1), SEG_B_LABEL, dtype=torch.float32)

    seg_a_with_label = torch.cat([seg_a_tensor, labels_a], dim=-1)
    seg_b_with_label = torch.cat([seg_b_tensor, labels_b], dim=-1)

    # Add 5th channel (affinity) if requested - 0 for segment points
    if affinity_channel:
        zeros_a = torch.zeros((n_a, 1), dtype=torch.float32)
        zeros_b = torch.zeros((n_b, 1), dtype=torch.float32)
        seg_a_with_label = torch.cat([seg_a_with_label, zeros_a], dim=-1)
        seg_b_with_label = torch.cat([seg_b_with_label, zeros_b], dim=-1)

    return seg_a_with_label, seg_b_with_label


def contact_faces_to_tensor(
    contact_faces: np.ndarray,
    contact_label: float | None = 0.0,
    affinity_channel_mode: str | None = None,
) -> torch.Tensor | None:
    """Convert contact faces to tensor with proper labeling.

    Args:
        contact_faces: Contact faces array, shape [N, 4] where 4th col is affinity.
        contact_label: Label for contact face points (4th channel). If None,
            uses affinity values from contact_faces.
        affinity_channel_mode: Mode for 5th affinity channel:
            - None: No 5th channel (4-channel output)
            - "per_point": 5th channel has per-point affinity values
            - "mean": 5th channel has mean affinity broadcast to all points

    Returns:
        Tensor of shape [N_valid, 4] or [N_valid, 5], or None if no valid faces.
    """
    cf_tensor = torch.tensor(contact_faces, dtype=torch.float32)

    # Filter out zero-padded contact faces
    valid_mask = torch.any(cf_tensor[:, :3] != 0, dim=1)
    valid_cf = cf_tensor[valid_mask]

    if valid_cf.shape[0] == 0:
        return None

    # Get per-point affinities before potentially overwriting
    per_point_affinities = valid_cf[:, 3].clone()

    if contact_label is not None:
        # Replace 4th channel with contact_label
        cf_with_label = valid_cf.clone()
        cf_with_label[:, 3] = contact_label
    else:
        # Keep affinity as 4th channel
        cf_with_label = valid_cf.clone()

    # Add 5th channel (affinity) if requested
    if affinity_channel_mode is not None:
        n_cf = cf_with_label.shape[0]
        if affinity_channel_mode == "per_point":
            aff_channel = per_point_affinities.unsqueeze(-1)
        elif affinity_channel_mode == "mean":
            mean_aff = per_point_affinities.mean()
            aff_channel = torch.full((n_cf, 1), mean_aff, dtype=torch.float32)
        else:
            raise ValueError(
                f"Unknown affinity_channel_mode: {affinity_channel_mode}. "
                "Must be 'per_point', 'mean', or None."
            )
        cf_with_label = torch.cat([cf_with_label, aff_channel], dim=-1)

    return cf_with_label


def contact_to_tensor(
    contact: SegContact,
    config_key: tuple[int, int],
    include_contact_faces: bool = False,
    contact_label: float | None = 0.0,
    affinity_channel_mode: str | None = None,
) -> torch.Tensor | None:
    """Convert a single contact to a pointcloud tensor.

    This function handles the full conversion including segment labeling,
    contact faces, and affinity channel. Used by both training and inference.

    Args:
        contact: SegContact object with local_pointclouds.
        config_key: Specific (radius_nm, n_points) config to use.
        include_contact_faces: If True, append contact_faces as additional points.
        contact_label: Label for contact face points (4th channel). If None,
            uses affinity values from contact_faces.
        affinity_channel_mode: Mode for optional 5th affinity channel:
            - None: No 5th channel (4-channel output)
            - "per_point": 5th channel has per-point affinity for CF, 0 for segments
            - "mean": 5th channel has mean affinity for CF, 0 for segments

    Returns:
        Tensor of shape [N, 4] or [N, 5] where N is total points, or None if
        contact doesn't have the required pointcloud config.
    """
    if contact.local_pointclouds is None:
        return None

    pc_data = contact.local_pointclouds.get(config_key)
    if pc_data is None:
        return None

    seg_a_pts = pc_data.get(contact.seg_a)
    seg_b_pts = pc_data.get(contact.seg_b)

    if seg_a_pts is None or seg_b_pts is None:
        return None

    # Use shared labeling function
    seg_a_with_label, seg_b_with_label = pointcloud_to_labeled_tensor(
        seg_a_pts, seg_b_pts, affinity_channel=affinity_channel_mode is not None
    )

    contact_pointcloud_parts = [seg_a_with_label, seg_b_with_label]

    # Optionally include contact faces
    if include_contact_faces and contact.contact_faces.shape[0] > 0:
        cf_tensor = contact_faces_to_tensor(
            contact.contact_faces,
            contact_label=contact_label,
            affinity_channel_mode=affinity_channel_mode,
        )
        if cf_tensor is not None:
            contact_pointcloud_parts.append(cf_tensor)

    return torch.cat(contact_pointcloud_parts, dim=0)


def contacts_to_tensor(
    contacts: Sequence[SegContact],
    config_key: tuple[int, int] | None = None,
    include_contact_faces: bool = False,
    contact_label: float | None = 0.0,
    affinity_channel_mode: str | None = None,
) -> tuple[torch.Tensor, list[int]]:
    """Convert contacts to batched pointcloud tensor for model input.

    Extracts pointclouds from contacts and converts them to a tensor format
    suitable for PointNet-style models. Uses segment labels -1 for seg_a and
    +1 for seg_b to match the training convention.

    IMPORTANT: The parameters must match the validation dataset configuration
    used during training to ensure consistent data representation.

    Args:
        contacts: Sequence of SegContact objects with local_pointclouds.
        config_key: Specific (radius_nm, n_points) config to use. If None,
            uses the first available config from sorted keys.
        include_contact_faces: If True, append contact_faces as additional
            points in the pointcloud.
        contact_label: Label for contact face points (4th channel). If None,
            uses affinity values from contact_faces.
        affinity_channel_mode: Mode for optional 5th affinity channel:
            - None: No 5th channel (4-channel output)
            - "per_point": 5th channel has per-point affinity for CF, 0 for segments
            - "mean": 5th channel has mean affinity for CF, 0 for segments

    Returns:
        Tuple of:
        - pointcloud: Tensor of shape [B, 4 or 5, N] where B is number of valid
          contacts and N is total points. Channel 0-2 are xyz, channel 3 is
          segment label (-1 or +1) or contact_label for CF points.
        - valid_indices: List of indices into original contacts sequence for
          contacts that had valid pointclouds.

    Example:
        >>> contacts = layer[idx]  # Read contacts from layer
        >>> tensor, indices = contacts_to_tensor(contacts)
        >>> probs = model(tensor)  # Run inference
        >>> for i, prob in zip(indices, probs):
        ...     contacts[i].merge_probabilities["authority"] = prob.item()
    """
    pointclouds_list: list[torch.Tensor] = []
    valid_indices: list[int] = []

    # Determine config key from first contact if not specified
    if config_key is None:
        for contact in contacts:
            if contact.local_pointclouds:
                config_key = sorted(contact.local_pointclouds.keys())[0]
                break

    if config_key is None:
        # No contacts have pointclouds
        return torch.zeros((0, 4, 0), dtype=torch.float32), []

    for idx, contact in enumerate(contacts):
        tensor = contact_to_tensor(
            contact, config_key, include_contact_faces, contact_label, affinity_channel_mode
        )
        if tensor is not None:
            pointclouds_list.append(tensor)
            valid_indices.append(idx)

    if not pointclouds_list:
        return torch.zeros((0, 4, 0), dtype=torch.float32), []

    # Stack and transpose to [B, 4, N]
    batched = torch.stack(pointclouds_list, dim=0)  # [B, N, 4]
    batched = batched.transpose(1, 2)  # [B, 4, N]

    return batched, valid_indices
