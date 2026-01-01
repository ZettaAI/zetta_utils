from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import attrs
import torch

from zetta_utils import builder
from zetta_utils.geometry import BBox3D
from zetta_utils.layer.volumetric.seg_contact import SegContactLayerBackend

from .sample_indexers import SegContactIndexer


@builder.register("SegContactDataset")
@attrs.frozen
class SegContactDataset(torch.utils.data.Dataset):
    """PyTorch dataset for seg_contact data.

    Returns dict of tensors for each contact, suitable for training merge classifiers.

    :param backend: SegContactLayerBackend to read from.
    :param bbox: Optional bounding box to restrict contacts.
    :param resolution: Resolution for the bounding box. Required if bbox is provided.
    :param pointcloud_config: Config key for local pointclouds (e.g., "r500_n64").
        If None, pointclouds are not included.
    :param merge_decision_authority: Authority name for merge decisions (e.g., "human").
        If None, target is not included.
    """

    backend: SegContactLayerBackend
    bbox: BBox3D | None = None
    resolution: Sequence[float] | None = None
    pointcloud_config: str | None = None
    merge_decision_authority: str | None = None

    _indexer: SegContactIndexer = attrs.field(init=False)

    def __attrs_pre_init__(self):
        super().__init__()

    def __attrs_post_init__(self):
        indexer = SegContactIndexer(
            backend=self.backend,
            bbox=self.bbox,
            resolution=self.resolution,
        )
        object.__setattr__(self, "_indexer", indexer)

    def __len__(self) -> int:
        return len(self._indexer)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        contact = self._indexer(idx)

        result: dict[str, Any] = {}

        # Core contact info
        result["contact_id"] = torch.tensor(contact.id, dtype=torch.int64)
        result["seg_a"] = torch.tensor(contact.seg_a, dtype=torch.int64)
        result["seg_b"] = torch.tensor(contact.seg_b, dtype=torch.int64)
        result["com"] = torch.tensor(list(contact.com), dtype=torch.float32)
        result["contact_faces"] = torch.tensor(contact.contact_faces, dtype=torch.float32)

        # Pointclouds if available and requested
        if self.pointcloud_config is not None and contact.local_pointclouds is not None:
            pc_data = contact.local_pointclouds.get(self.pointcloud_config)
            if pc_data is not None:
                seg_a_pts = pc_data.get(contact.seg_a)
                seg_b_pts = pc_data.get(contact.seg_b)
                if seg_a_pts is not None:
                    result["pointcloud_a"] = torch.tensor(seg_a_pts, dtype=torch.float32)
                if seg_b_pts is not None:
                    result["pointcloud_b"] = torch.tensor(seg_b_pts, dtype=torch.float32)

        # Merge decision target if available and requested
        if self.merge_decision_authority is not None and contact.merge_decisions is not None:
            should_merge = contact.merge_decisions.get(self.merge_decision_authority)
            if should_merge is not None:
                result["target"] = torch.tensor(1.0 if should_merge else 0.0, dtype=torch.float32)

        # Partner metadata as-is (can be any type)
        if contact.partner_metadata is not None:
            meta_a = contact.partner_metadata.get(contact.seg_a)
            meta_b = contact.partner_metadata.get(contact.seg_b)
            if meta_a is not None:
                result["metadata_a"] = meta_a
            if meta_b is not None:
                result["metadata_b"] = meta_b

        return result
