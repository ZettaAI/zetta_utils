from __future__ import annotations

import json
from typing import Any

import attrs
import fsspec
import numpy as np

from zetta_utils.geometry import Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex


def read_info(path: str) -> dict:
    """Read seg_contact info file from path."""
    info_path = f"{path}/info"
    fs, fs_path = fsspec.core.url_to_fs(info_path)
    if not fs.exists(fs_path):
        raise FileNotFoundError(f"Info file not found: {info_path}")
    with fs.open(fs_path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


@attrs.mutable
class SegContact:
    """Represents a contact interface between two segments."""

    id: int
    seg_a: int
    seg_b: int
    com: Vec3D[float]  # center of mass in nm
    contact_faces: np.ndarray  # (N, 4) float32: x, y, z, affinity in nm
    representative_points: dict[int, Vec3D[float]]  # segment_id -> point in nm (required)
    representative_supervoxels: dict[int, int] | None = (
        None  # segment_id -> supervoxel_id (uint64)
    )
    # (radius_nm, n_points) -> {segment_id -> (n_points, 3) in nm}
    local_pointclouds: dict[tuple[int, int], dict[int, np.ndarray]] | None = None
    merge_decisions: dict[str, bool] | None = None  # authority -> yes/no
    merge_probabilities: dict[str, float] | None = None  # authority -> probability [0.0, 1.0]
    partner_metadata: dict[int, Any] | None = None  # segment_id -> metadata
    # Original contact_faces coordinates in nm (preserved before normalization)
    contact_faces_original_nm: np.ndarray | None = None

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
