from __future__ import annotations

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


@builder.register("SegContactDataset")
@attrs.frozen
class SegContactDataset(torch.utils.data.Dataset):
    """PyTorch dataset for seg_contact data - returns chunk-aligned batches.

    Uses layer + indexer architecture:
    - Indexer returns VolumetricIndex (bounding box) for each chunk
    - Layer reads contacts via read_with_procs (enabling augmentations/normalization)

    Outputs concatenated pointclouds from all available configs in sorted order.
    Each pointcloud has xyz + 4th channel:
    - For local pointclouds: 4th channel is segment_label (0 for seg_a, 1 for seg_b)
    - For contact_faces: 4th channel is affinity value (0.0 to 1.0)

    :param layer: VolumetricSegContactLayer to read contacts from.
    :param sample_indexer: Indexer that returns VolumetricIndex per chunk.
    :param merge_decision_authority: Authority name for merge decisions.
    :param max_contact_faces: Fixed size for contact_faces tensor (pad/truncate).
    :param include_contact_faces_in_pointcloud: If True, append contact_faces as
        additional points in the pointcloud (with affinity as 4th channel).
    """

    layer: VolumetricSegContactLayer
    sample_indexer: SampleIndexer
    merge_decision_authority: str | None = None
    max_contact_faces: int = 2048
    include_contact_faces_in_pointcloud: bool = False

    def __attrs_pre_init__(self):
        super().__init__()

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
        pointclouds_list = []  # List of [n_points, 4] tensors per contact
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

                # Add segment labels: 0 for seg_a, 1 for seg_b
                seg_a_tensor = torch.tensor(seg_a_pts, dtype=torch.float32)
                seg_b_tensor = torch.tensor(seg_b_pts, dtype=torch.float32)

                n_a, n_b = seg_a_tensor.shape[0], seg_b_tensor.shape[0]

                labels_a = torch.zeros(n_a, 1, dtype=torch.float32)
                labels_b = torch.ones(n_b, 1, dtype=torch.float32)

                seg_a_with_label = torch.cat([seg_a_tensor, labels_a], dim=-1)
                seg_b_with_label = torch.cat([seg_b_tensor, labels_b], dim=-1)

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

            # Optionally include contact faces as additional points
            if self.include_contact_faces_in_pointcloud:
                # contact_faces is [max_faces, 4] where columns are x, y, z, affinity
                # Keep xyz + affinity (4th channel is affinity, not segment label)
                contact_pointcloud_parts.append(contact_faces_tensor)

            if contact_pointcloud_parts:
                contact_pointcloud = torch.cat(contact_pointcloud_parts, dim=0)
                pointclouds_list.append(contact_pointcloud)

            # Target
            if self.merge_decision_authority is not None and contact.merge_decisions is not None:
                should_merge = contact.merge_decisions.get(self.merge_decision_authority)
                if should_merge is not None:
                    targets.append(1.0 if should_merge else 0.0)

        result: dict[str, Any] = {
            "contact_id": torch.tensor(contact_ids, dtype=torch.int64),
            "seg_a": torch.tensor(seg_as, dtype=torch.int64),
            "seg_b": torch.tensor(seg_bs, dtype=torch.int64),
            "com": torch.tensor(coms, dtype=torch.float32),
            "contact_faces": torch.stack(contact_faces_list),
        }

        if pointclouds_list:
            result["pointcloud"] = torch.stack(pointclouds_list)  # [B, total_points, 4]
        if targets:
            result["target"] = torch.tensor(targets, dtype=torch.float32)

        return result
