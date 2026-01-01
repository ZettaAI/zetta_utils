from __future__ import annotations

import json
from typing import Any

import attrs
import numpy as np
from cloudfiles import CloudFile

from zetta_utils.geometry import Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex


def read_info(path: str) -> dict:
    """Read seg_contact info file from path."""
    info_path = f"{path}/info"
    cf = CloudFile(info_path)
    if not cf.exists():
        raise FileNotFoundError(f"Info file not found: {info_path}")
    return json.loads(cf.get().decode("utf-8"))


@attrs.mutable
class SegContact:
    """Represents a contact interface between two segments."""

    id: int
    seg_a: int
    seg_b: int
    com: Vec3D[float]  # center of mass in nm
    contact_faces: np.ndarray  # (N, 4) float32: x, y, z, affinity in nm
    # config_key -> {segment_id -> (n_points, 3) in nm}
    # config_key format: "r{radius_nm}_n{n_points}" e.g. "r1000_n2048"
    local_pointclouds: dict[str, dict[int, np.ndarray]] | None = None
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
