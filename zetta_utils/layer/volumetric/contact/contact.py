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
        raise NotImplementedError

    def with_converted_coordinates(
        self, from_res: Vec3D, to_res: Vec3D
    ) -> Contact:
        """Return new Contact with coordinates converted between resolutions."""
        raise NotImplementedError
