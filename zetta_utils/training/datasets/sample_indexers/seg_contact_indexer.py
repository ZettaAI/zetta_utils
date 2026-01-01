from __future__ import annotations

from bisect import bisect_right
from typing import Sequence

import attrs

from zetta_utils import builder
from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex
from zetta_utils.layer.volumetric.seg_contact import SegContact, SegContactLayerBackend

from .base import SampleIndexer


@builder.register("SegContactIndexer")
@attrs.frozen
class SegContactIndexer(SampleIndexer[SegContact]):
    """SampleIndexer for seg_contact data.

    Enumerates all contacts within a bounding box and provides access by index.
    Uses get_contact_counts for efficient enumeration without loading all data.
    Contacts are loaded lazily when accessed.

    :param backend: SegContactLayerBackend to read from.
    :param bbox: Optional bounding box to restrict contacts. If None, uses full layer bounds.
    :param resolution: Resolution for the bounding box. Required if bbox is provided.
    """

    backend: SegContactLayerBackend
    bbox: BBox3D | None = None
    resolution: Sequence[float] | None = None

    # Computed at init - only counts, not actual contacts
    _chunk_indices: list[tuple[int, int, int]] = attrs.field(init=False, factory=list)
    _chunk_counts: list[int] = attrs.field(init=False, factory=list)
    _cumulative_counts: list[int] = attrs.field(init=False, factory=list)
    _total_count: int = attrs.field(init=False, default=0)
    _idx: VolumetricIndex = attrs.field(init=False)

    def __attrs_post_init__(self):
        # Determine the bounding box to use
        if self.bbox is not None:
            if self.resolution is None:
                raise ValueError("resolution is required when bbox is provided")
            bbox = self.bbox
            resolution = Vec3D(*self.resolution)
        else:
            # Use full layer bounds from backend
            start_voxel = self.backend.voxel_offset
            end_voxel = Vec3D(
                start_voxel[0] + self.backend.size[0],
                start_voxel[1] + self.backend.size[1],
                start_voxel[2] + self.backend.size[2],
            )
            bbox = BBox3D.from_coords(
                start_coord=list(start_voxel),
                end_coord=list(end_voxel),
                resolution=list(self.backend.resolution),
            )
            resolution = Vec3D[float](*self.backend.resolution)

        # Create VolumetricIndex for reading
        idx = VolumetricIndex(bbox=bbox, resolution=resolution)
        object.__setattr__(self, "_idx", idx)

        # Get contact counts per chunk (doesn't load actual contacts)
        chunk_counts_dict = self.backend.get_contact_counts(idx)

        # Build ordered lists for O(log n) lookup
        chunk_indices = []
        chunk_counts = []
        cumulative_counts = []
        total = 0

        for chunk_idx in sorted(chunk_counts_dict.keys()):
            count = chunk_counts_dict[chunk_idx]
            if count > 0:
                chunk_indices.append(chunk_idx)
                chunk_counts.append(count)
                cumulative_counts.append(total)
                total += count

        object.__setattr__(self, "_chunk_indices", chunk_indices)
        object.__setattr__(self, "_chunk_counts", chunk_counts)
        object.__setattr__(self, "_cumulative_counts", cumulative_counts)
        object.__setattr__(self, "_total_count", total)

    def __len__(self) -> int:
        return self._total_count

    def __call__(self, idx: int) -> SegContact:
        """Get contact at the given index.

        Lazily loads the chunk containing this contact.

        :param idx: Contact index (0 to len-1).
        :return: SegContact at that index.
        """
        if idx < 0 or idx >= self._total_count:
            raise IndexError(f"Index {idx} out of range [0, {self._total_count})")

        # Binary search to find which chunk contains this index
        chunk_pos = max(bisect_right(self._cumulative_counts, idx) - 1, 0)

        chunk_idx = self._chunk_indices[chunk_pos]
        local_idx = idx - self._cumulative_counts[chunk_pos]

        # Load just this chunk and get the contact
        # Filter contacts to those within bounds
        contacts = self.backend.read_chunk(chunk_idx)
        start_nm = self._idx.bbox.start
        end_nm = self._idx.bbox.end

        in_bounds_contacts = [
            c
            for c in contacts
            if (
                start_nm[0] <= c.com[0] < end_nm[0]
                and start_nm[1] <= c.com[1] < end_nm[1]
                and start_nm[2] <= c.com[2] < end_nm[2]
            )
        ]

        return in_bounds_contacts[local_idx]


@builder.register("build_seg_contact_indexer")
def build_seg_contact_indexer(
    backend: SegContactLayerBackend,
    bbox: BBox3D | None = None,
    resolution: Sequence[float] | None = None,
) -> SegContactIndexer:
    """Build a SegContactIndexer.

    :param backend: SegContactLayerBackend to read from.
    :param bbox: Optional bounding box to restrict contacts.
    :param resolution: Resolution for the bounding box.
    :return: SegContactIndexer instance.
    """
    return SegContactIndexer(backend=backend, bbox=bbox, resolution=resolution)
