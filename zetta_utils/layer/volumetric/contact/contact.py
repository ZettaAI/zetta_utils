from __future__ import annotations

from typing import Any

import attrs
import numpy as np

from zetta_utils.geometry import Vec3D, BBox3D
from zetta_utils.layer.volumetric import VolumetricIndex


@attrs.frozen
class Contact:
    """Represents a contact interface between two segments."""

    id: int
    seg_a: int
    seg_b: int
    com: Vec3D[float]  # center of mass in nm
    contact_faces: np.ndarray  # (N, 4) float32: x, y, z, affinity in nm
    local_pointclouds: dict[int, np.ndarray] | None = None  # segment_id -> (n_points, 3) in nm
    merge_decisions: dict[str, bool] | None = None  # authority -> yes/no
    partner_metadata: dict[int, Any] | None = None  # segment_id -> metadata

    def in_bounds(self, idx: VolumetricIndex) -> bool:
        """Check if COM falls within the given volumetric index."""
        bbox = idx.bbox
        # Convert bbox to nm
        start_nm = (
            bbox.start[0] * idx.resolution[0],
            bbox.start[1] * idx.resolution[1],
            bbox.start[2] * idx.resolution[2],
        )
        end_nm = (
            bbox.end[0] * idx.resolution[0],
            bbox.end[1] * idx.resolution[1],
            bbox.end[2] * idx.resolution[2],
        )
        return (
            start_nm[0] <= self.com[0] < end_nm[0]
            and start_nm[1] <= self.com[1] < end_nm[1]
            and start_nm[2] <= self.com[2] < end_nm[2]
        )

    def with_converted_coordinates(
        self, from_res: Vec3D, to_res: Vec3D
    ) -> Contact:
        """Return new Contact with coordinates converted between resolutions.

        Note: Contact coordinates are stored in nanometers, so resolution
        conversion doesn't change the values - this method exists for API
        consistency with other layer types.
        """
        # Coordinates are in nm, they don't change with resolution
        # Just return a copy with same values
        return Contact(
            id=self.id,
            seg_a=self.seg_a,
            seg_b=self.seg_b,
            com=self.com,
            contact_faces=self.contact_faces.copy(),
            local_pointclouds=(
                {k: v.copy() for k, v in self.local_pointclouds.items()}
                if self.local_pointclouds is not None
                else None
            ),
            merge_decisions=self.merge_decisions,
            partner_metadata=self.partner_metadata,
        )
