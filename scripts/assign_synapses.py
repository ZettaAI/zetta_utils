"""
Test script that manually loads and applies the assignment model
to a cutout (ROI) of ant data.  This involves:

  1. Loading the image, synapse segmentation, and cell segmentation layers for the ROI.
  2. Loading the synapse assignment model (in ONNX format).
  3. Iterating over the synapses, finding the centroid of each.
  4. Extracting a much smaller (24x24x8) cutout around each synapse.
  5. Running the model on this smaller cutout, resulting in two output masks:
     one for the presynaptic side, and one for the postsynaptic side.
  6. Finding the cell segmentation ID that most overlaps each mask, within
     a dilated region around the synapse segmentation.

We can write those results out in CSV format for now, but at some point we'll
also want to create a NG annotation layer with a line segment for each synapse.
To do that, we'll need to find a 3D point that best represents the pre- and
post-synaptic side of each synapse.
"""

from collections import deque
from datetime import datetime

# Imports
from math import floor
from typing import Any, Literal, Optional, Sequence, Tuple

import numpy as np
from check_model import load_model
from numpy.typing import NDArray
from scipy.stats import mode

import zetta_utils.tensor_ops.convert as convert
from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.precomputed import PrecomputedInfoSpec, PrecomputedVolumeDType
from zetta_utils.layer.volumetric import VolumetricIndex, VolumetricLayer
from zetta_utils.layer.volumetric.cloudvol import build_cv_layer

# Configuration
resolution = Vec3D(8, 8, 42)  # working resolution
image_path = "gs://dkronauer-ant-001-alignment-final/aligned"
image_res = Vec3D(8, 8, 42)
synseg_path = "gs://dkronauer-ant-001-manual-labels/synapses/664d2b27010000cb00388fed/postsynaptic-terminal/000"
synseg_res = Vec3D(8, 8, 42)
cellseg_path = "gs://dkronauer-ant-001-kisuk/test/240507-finetune-z1400-3400/seg"
cellseg_res = Vec3D(16, 16, 42)

bounds_min = [28848, 18607, 3070]
bounds_max = [29104, 18863, 3110]

model_path = "gs://murthy_fly_001_syn_temp/assignment/experiments/jabae-fafb-assignment-exp-0601/models/model300000.onnx"

# Utility functions


def make_layer(
    bbox: BBox3D,
    resolution: Vec3D,
    path: str,
    data_type: PrecomputedVolumeDType = "int32",
    chunk_sizes: Sequence[Sequence[int]] = [[256, 256, 32]],
) -> VolumetricLayer:
    return build_cv_layer(
        path=path,
        info_overwrite=True,
        index_resolution=resolution,
        info_type="image",
        info_data_type=data_type,
        info_num_channels=1,
        info_encoding="raw",
        info_scales=[resolution],
        info_chunk_size=chunk_sizes[0],
        info_bbox=bbox,
        cv_kwargs={"non_aligned_writes": True},
    )


def load_volume(path: str, scale_index: int = 0) -> Tuple[VolumetricLayer, BBox3D]:
    """
    Load a CloudVolume given the path, and optionally, which scale (resolution) is desired.
    Return the CloudVolume, and a BBox3D describing the data bounds.
    """
    spec = PrecomputedInfoSpec(info_path=path)
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


def bfs_nearest_value(
    array: np.ndarray, start: Sequence[int], target_value: int, xy_only: bool = False
) -> Tuple[int, int, int] | None:
    """
    Perform a breadth-first search starting at the given point, to find
    the closest point in the array with the given value.

    Returns: (x, y, z) location of the point found, or None.
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
                    if neighbor_value == target_value:
                        return (nx, ny, nz)  # Found it!
                    queue.append(neighbor_pos)
                    visited.add(neighbor_pos)
    return None  # None found


# 1. Load the image, synapse segmentation, and cell segmentation layers for the ROI.
print(f"Working on voxel range {bounds_min} to {bounds_max}")
extent = list(Vec3D(*bounds_max) - Vec3D(*bounds_min))
print(
    f"Extent: [{extent[0]}, {extent[1]}, {extent[2]}],"
    f"  Total: {(extent[0])*(extent[1])*(extent[2])} voxels"
)

print(f"Loading synapse segmentation: {synseg_path}")
synseg_layer, bounds = load_volume(synseg_path)
synseg_data = synseg_layer[
    resolution,
    bounds.start.x : bounds.end.x,
    bounds.start.y : bounds.end.y,
    bounds.start.z : bounds.end.z,
]
synseg_data = synseg_data[0]  # (use only channel 0)

print(f"Loading image layer: {image_path}")
image_layer = build_cv_layer(
    path=image_path,
    allow_slice_rounding=True,
    index_resolution=resolution,
    data_resolution=image_res,
    interpolation_mode="nearest",
)
image_data = image_layer[
    resolution,
    bounds_min[0] : bounds_max[0],
    bounds_min[1] : bounds_max[1],
    bounds_min[2] : bounds_max[2],
]
image_data = image_data[0]  # (use only channel 0)

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
    bounds_min[0] : bounds_max[0],
    bounds_min[1] : bounds_max[1],
    bounds_min[2] : bounds_max[2],
]
cellseg_data = cellseg_data[0]  # (use only channel 0)

# 1b. Prepare output layers as needed.
nowstr = datetime.now().strftime("%Y%m%d%H%M%S")  # new file on every run
nowstr = datetime.now().strftime("%Y%m%d")  # new file daily (but same within each day)
outsynmask_path = f"gs://dkronauer-ant-001-synapse/test/js/{nowstr}/synmask"
outpre_path = f"gs://dkronauer-ant-001-synapse/test/js/{nowstr}/pre"
outpost_path = f"gs://dkronauer-ant-001-synapse/test/js/{nowstr}/post"
bbox = BBox3D.from_coords(Vec3D(*bounds_min), Vec3D(*bounds_max), unit="voxels")
print(f"Creating output layer: {outsynmask_path}")
outsynmask_layer = make_layer(bbox, resolution, outsynmask_path, "int32")
outsynmask_data = np.zeros(extent, np.int32)
print(f"Creating output layer: {outpre_path}")
outpre_layer = make_layer(bbox, resolution, outpre_path, "float32")
outpre_data = np.zeros(extent, np.float32)
print(f"Creating output layer: {outpost_path}")
outpost_layer = make_layer(bbox, resolution, outpost_path, "float32")
outpost_data = np.zeros(extent, np.float32)

# 2. Load the synapse assignment model (in ONNX format, as an ONNX inference session).
print(f"Loading assignment model: {model_path}")
model = load_model(model_path)
model_input = model.get_inputs()[0]
input_name = model_input.name
input_shape = model_input.shape
window_size = [input_shape[4], input_shape[3], input_shape[2]]
print(f"Model input shape: {input_shape}; Window size: {window_size}; type {model_input.type}")

# 3. Iterate over synapses, finding centroid of each.
syn_ids = np.unique(synseg_data)
syn_ids = syn_ids[syn_ids != 0]
# syn_ids = [32, 55]; print(f"HACK: only doing {syn_ids}")
print(f"This region contains {len(syn_ids)} synapses")
csv_file = open("syn_inf.csv", "w")
csv_file.write("Synapse ID,Presynaptic Cell ID,Postsynaptic Cell ID\n")

annotations = []
annotations.append('"annotations": [')

# NG has an annoying habit of displaying half a Z unit up from what it claims
# to be displaying.  As a result, if your Z slices are thick, you can't see any
# annotations on nice round Z values.  Work around this by adding:
pos_offset = Vec3D(0, 0, 0.5)

for syn_id in syn_ids:
    coords = np.argwhere(synseg_data == syn_id)
    centroid = centroid_of_id(synseg_data, syn_id)
    print(f"Synapse {syn_id}: {len(coords)} voxels with centroid {list(centroid)}...")

    # 4. Extract small window around the synapse in our various layers.
    xmin = floor(centroid[0] - window_size[0] / 2)
    ymin = floor(centroid[1] - window_size[1] / 2)
    zmin = floor(centroid[2] - window_size[2] / 2)
    xmax = xmin + window_size[0]
    ymax = ymin + window_size[1]
    zmax = zmin + window_size[2]
    if (
        xmin < 0
        or ymin < 0
        or zmin < 0
        or xmax >= image_data.shape[0]
        or ymax >= image_data.shape[1]
        or zmax >= image_data.shape[2]
    ):
        print(f"Window is out of bounds; skipping this one.")
        continue
    image_wind = image_data[xmin:xmax, ymin:ymax, zmin:zmax]
    synseg_wind = synseg_data[xmin:xmax, ymin:ymax, zmin:zmax]
    cellseg_wind = cellseg_data[xmin:xmax, ymin:ymax, zmin:zmax]

    # 5. Run the model on this window.
    syn_mask = np.where(synseg_wind == syn_id, 1, 0)
    input_tensor = np.stack(
        [  # Reshape inputs (8, 24, 24) -> (2, 8, 24, 24)
            np.transpose(image_wind, (2, 0, 1)),
            np.transpose(syn_mask, (2, 0, 1)),
        ]
    )
    input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension to the input densor
    input_tensor = input_tensor.astype(np.float32)

    output = model.run(None, {input_name: input_tensor})
    output_array = output[0]

    # print("Synapse mask:")
    # print(np.transpose(syn_mask, (2,0,1))[4,:,:])
    # out = (output_array * 100).astype(int)
    # print("Output channel 0:")
    # print(out[0,0,4,:,:])
    # print("Output channel 1:")
    # print(out[0,1,4,:,:])

    # transpose to match the pattern of our image data
    presyn_output = np.transpose(output_array[0, 0], (1, 2, 0))
    postsyn_output = np.transpose(output_array[0, 1], (1, 2, 0))

    # (Add to output data, just for debugging purposes)
    if syn_id in syn_ids:  # [13, 32, 51, 69, 84, 100, 114, 139, 191, 204, 211]:
        outpre_data[xmin:xmax, ymin:ymax, zmin:zmax] += presyn_output
        outpost_data[xmin:xmax, ymin:ymax, zmin:zmax] += postsyn_output
        print(
            f"Wrote inference data around {syn_id} to output at {list(bounds.start + Vec3D(*centroid))}"
        )

    # 6. Find the cell ID that maximizes the presynaptic output.
    best_sum = 0
    pre_cell_id: int | None = None
    candidate_cell_ids = np.unique(cellseg_wind)  # (or could be limited by dilated synapse mask)
    for cell_id in candidate_cell_ids:
        cell_mask = np.where(cellseg_wind == cell_id, 1, 0)
        sum = np.sum(cell_mask * presyn_output)
        if sum > best_sum:
            best_sum = sum
            pre_cell_id = cell_id
    assert pre_cell_id is not None

    # ...and to find the postsynaptic cell ID, just take the most common value (i.e. mode)
    # in the cell ID window under the synapse mask.
    post_cell_id = mode(cellseg_wind[syn_mask.astype(bool)])[0]
    print(f"Synapse: {syn_id}  Pre: {pre_cell_id}  Post: {post_cell_id}")
    csv_file.write(f"{syn_id},{pre_cell_id},{post_cell_id}\n")

    pre_cell_point = bfs_nearest_value(
        cellseg_wind, tuple(i // 2 for i in window_size), pre_cell_id, True
    )
    if pre_cell_point is None:
        pre_cell_point = bfs_nearest_value(
            cellseg_wind, tuple(i // 2 for i in window_size), pre_cell_id, False
        )
    assert pre_cell_point is not None
    pre_cell_point = (xmin + pre_cell_point[0], ymin + pre_cell_point[1], zmin + pre_cell_point[2])
    annotations.append("{")
    annotations.append(f'"pointA": {list(bounds.start + Vec3D(*pre_cell_point) + pos_offset)},')
    annotations.append(f'"pointB": {list(bounds.start + Vec3D(*centroid) + pos_offset)},')
    annotations.append('"type": "line",')
    annotations.append(f'"description": "{pre_cell_id} to {post_cell_id}",')
    annotations.append(f'"id": "{syn_id}"')
    annotations.append("}," if syn_id != syn_ids[-1] else "}")
annotations.append("],")

print("\nNG annotations:\n")
print("\n".join(annotations))

csv_file.close()

# Save data to output layers
outpre_data = np.clip(outpre_data, 0.0, 1.0)
outpost_data = np.clip(outpost_data, 0.0, 1.0)
outpre_layer[
    resolution,
    bounds_min[0] : bounds_max[0],
    bounds_min[1] : bounds_max[1],
    bounds_min[2] : bounds_max[2],
] = np.expand_dims(outpre_data, axis=0)
print(f"Wrote data centered on {np.array(bounds_min) + centroid} to {outpre_path}")
outpost_layer[
    resolution,
    bounds_min[0] : bounds_max[0],
    bounds_min[1] : bounds_max[1],
    bounds_min[2] : bounds_max[2],
] = np.expand_dims(outpost_data, axis=0)
