"""
This script tries a much simpler approach to synaptic assignment for the case where
we have identified the postsynaptic terminal rather than the cleft.  For each synapse:

1. Find the centroid.
2. Find the cell at that point; this is the postsynaptic partner.
3. Find the closest other cell to that point; this is the presynaptic partner.

This script outputs NG annotations that draw a little line between these two
points, with the cell IDs in the description, for easy visual checking.
"""

from collections import deque
from datetime import datetime

# Imports
from math import floor
from typing import Any, Literal, Optional, Sequence, Tuple

import nglui
import numpy as np
from caveclient import CAVEclient
from check_model import load_model
from numpy.typing import NDArray

import zetta_utils.tensor_ops.convert as convert
from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex, VolumetricLayer
from zetta_utils.layer.volumetric.cloudvol import build_cv_layer
from zetta_utils.layer.volumetric.precomputed import PrecomputedInfoSpec

# Configuration
resolution: Vec3D = Vec3D(8, 8, 42)  # working resolution
synseg_path: str = "precomputed://gs://dkronauer-ant-001-manual-labels/synapses/664d2b27010000cb00388fed/postsynaptic-terminal/000"
cellseg_path: str = "gs://dkronauer-ant-001-kisuk/test/240507-finetune-z1400-3400/seg"
cellseg_res: Vec3D = Vec3D(16, 16, 42)


def load_volume(path: str, scale_index: int = 0) -> Tuple[VolumetricLayer, BBox3D]:
    """
    Load a CloudVolume given the path, and optionally, which scale (resolution) is desired.
    Return the CloudVolume, and a BBox3D describing the data bounds.
    """
    spec = PrecomputedInfoSpec(reference_path=path)
    info = spec.make_info()
    assert info is not None
    scale = info["scales"][scale_index]
    resolution = scale["resolution"]
    start_coord = scale["voxel_offset"]
    size = scale["size"]
    end_coord = [a + b for (a, b) in zip(start_coord, size)]
    cvl = build_cv_layer(
        path=path,
        allow_slice_rounding=True,
        default_desired_resolution=resolution,
        index_resolution=resolution,
        data_resolution=resolution,
        interpolation_mode=info["type"],
    )
    bounds = BBox3D.from_coords(start_coord, end_coord)
    return cvl, bounds


def centroid_of_id(array: np.ndarray, id: int) -> NDArray[np.int_]:
    """
    Find the (rounded to int) centroid of locations in the array with a value equal to id.
    """
    coordinates = np.argwhere(array == id)
    assert coordinates.size > 0
    centroid = np.mean(coordinates, axis=0)
    return np.round(centroid).astype(int)


def bfs_nearest_different_value(
    array: np.ndarray, start: Sequence[int], xy_only: bool = False
) -> Tuple[Optional[Tuple[int, int, int]], Optional[int]]:
    """
    Perform a breadth-first search starting at the given point, to find
    the closest point in the array with a nonzero value different from
    the value at start.

    Returns: (location, value) of the point found, or (None, None).
    """
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]
    if not xy_only:
        directions += [(0, 0, 1), (0, 0, -1)]
    queue = deque([start])
    value_at_start = array[tuple(start)]
    visited = set()
    visited.add(tuple(start))

    while queue:
        x, y, z = queue.popleft()
        for dx, dy, dz in directions:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < array.shape[0] and 0 <= ny < array.shape[1] and 0 <= nz < array.shape[2]:
                neighbor_pos = (nx, ny, nz)
                if neighbor_pos not in visited:
                    neighbor_value = array[nx, ny, nz]
                    if neighbor_value != 0 and neighbor_value != value_at_start:
                        return (nx, ny, nz), neighbor_value  # Found it!
                    queue.append(neighbor_pos)
                    visited.add(neighbor_pos)
    return None, None  # None found


# Load the synapse segmentation and cell segmentation layers for the ROI
# (which is defined by the bounds of the synapse segmentation).

print(f"Loading synapse segmentation: {synseg_path}")
synseg_layer, bounds = load_volume(synseg_path)
synseg_data = synseg_layer[
    resolution,
    bounds.start.x : bounds.end.x,
    bounds.start.y : bounds.end.y,
    bounds.start.z : bounds.end.z,
]
synseg_data = synseg_data[0]  # (use only channel 0)

print(f"Working on voxel range {bounds.start} to {bounds.end}")
extent = list(bounds.end - bounds.start)
print(
    f"Extent: [{extent[0]}, {extent[1]}, {extent[2]}],"
    f"  Total: {(extent[0])*(extent[1])*(extent[2])} voxels"
)

print(f"Loading cell segmentation: {cellseg_path}")
cellseg_layer = build_cv_layer(
    path=cellseg_path,
    allow_slice_rounding=True,
    index_resolution=resolution,
    data_resolution=cellseg_res,
    interpolation_mode="nearest",
)
cellseg_data = cellseg_layer[
    resolution,
    bounds.start.x : bounds.end.x,
    bounds.start.y : bounds.end.y,
    bounds.start.z : bounds.end.z,
]
cellseg_data = cellseg_data[0]  # (use only channel 0)

# Iterate over the synapses, finding the closest other cell ID for each one.
synapse_ids: np.ndarray = np.unique(synseg_data[synseg_data != 0])
print(f"{len(synapse_ids)} synapses, ranging from {synapse_ids[0]} to {synapse_ids[-1]}")

annotations: list[str] = []
annotations.append('"annotations": [')

# NG has an annoying habit of displaying half a Z unit up from what it claims
# to be displaying.  As a result, if your Z slices are thick, you can't see any
# annotations on nice round Z values.  Work around this by adding:
csv_file = open("syn_simp.csv", "w")
csv_file.write("Synapse ID,Presynaptic Cell ID,Postsynaptic Cell ID\n")
pos_offset: Vec3D = Vec3D(0, 0, 0.5)
for id in synapse_ids:
    id_point = centroid_of_id(synseg_data, id)
    post_cell_id = cellseg_data[tuple(id_point)]
    pre_cell_point, pre_cell_id = bfs_nearest_different_value(cellseg_data, tuple(id_point), True)
    print(
        f"Synapse {id} at {id_point} is cell {post_cell_id}; partner at {pre_cell_point} is cell {pre_cell_id}"
    )
    csv_file.write(f"{id},{pre_cell_id},{post_cell_id}\n")

    annotations.append("{")
    annotations.append(f'"pointA": {list(bounds.start + Vec3D(*pre_cell_point) + pos_offset)},')  # type: ignore
    annotations.append(f'"pointB": {list(bounds.start + Vec3D(*id_point) + pos_offset)},')
    annotations.append('"type": "line",')
    annotations.append(f'"description": "{pre_cell_id} to {post_cell_id}",')
    annotations.append(f'"id": "{id}"')
    annotations.append("}," if id != synapse_ids[-1] else "}")
annotations.append("],")

print("\nNG annotations:\n")
print("\n".join(annotations))
