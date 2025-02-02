"""
This script prototypes a process for finding contacts between known presynaptic
cells in a smallish cutout, and any other cells in that cutout.
"""
import csv
import os
import sys
from datetime import datetime

import cc3d
import google.auth
import nglui
import numpy as np
from caveclient import CAVEclient
from scipy.ndimage import binary_dilation

from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex, VolumetricLayer
from zetta_utils.layer.volumetric.cloudvol import build_cv_layer
from zetta_utils.layer.volumetric.deprecated import PrecomputedInfoSpec

client: CAVEclient


def verify_cave_auth():
    global client
    client = CAVEclient()
    try:
        client.state
        return  # no exception?  All's good!
    except:
        pass
    print("Authentication needed.")
    print("Go to: https://global.daf-apis.com/auth/api/v1/create_token")
    token = input("Enter token: ")
    client.auth.save_token(token=token)


def input_or_default(prompt: str, value: str) -> str:
    response = input(f"{prompt} [{value}]: ")
    if response == "":
        response = value
    return response


def get_annotation_layer_name(state, label="synapse annotation"):
    names = nglui.parser.annotation_layers(state)
    if len(names) == 0:
        print("No annotation layers found in this state.")
        sys.exit()
    elif len(names) == 1:
        return names[0]
    while True:
        for i in range(0, len(names)):
            print(f"{i+1}. {names[i]}")
        choice: int | str = input(f"Enter {label} layer name or number: ")
        if choice in names:
            return choice
        choice = int(choice) - 1
        if choice >= 0 and choice < len(names):
            return names[choice]


def unarchived_segmentation_layers(state):
    names = nglui.parser.segmentation_layers(state)
    for i in range(len(names) - 1, -1, -1):
        if nglui.parser.get_layer(state, names[i]).get("archived", False):
            del names[i]
    return names


def get_segmentation_layer_name(state, label="cell segmentation"):
    names = unarchived_segmentation_layers(state)
    if len(names) == 0:
        print("No segmentation layers found in this state.")
        sys.exit()
    elif len(names) == 1:
        return names[0]
    while True:
        for i in range(0, len(names)):
            print(f"{i+1}. {names[i]}")
        choice: int | str = input(f"Enter {label} layer name or number: ")
        if choice in names:
            return choice
        choice = int(choice) - 1
        if choice >= 0 and choice < len(names):
            return names[choice]


def get_bounding_box(precomputed_path, scale_index=0):
    """
    Given a path to a precomputed volume, return a VolumetricIndex describing the data bounds
    and the data resolution.
    """
    spec = PrecomputedInfoSpec(reference_path=precomputed_path)
    disable_stdout()  # (following call spews Google warning sometimes)
    info = spec.make_info()
    enable_stdout()
    assert info is not None
    scale = info["scales"][scale_index]
    resolution = scale["resolution"]
    start_coord = scale["voxel_offset"]
    size = scale["size"]
    end_coord = [a + b for (a, b) in zip(start_coord, size)]
    bounds = VolumetricIndex.from_coords(start_coord, end_coord, resolution)
    return bounds


def load_volume(path, scale_index=0, index_resolution=None):
    """
    Load a CloudVolume given the path, and optionally, which scale (resolution) is desired.
    Return the CloudVolume, and a VolumetricIndex describing the data bounds and resolution.
    """
    spec = PrecomputedInfoSpec(reference_path=path)
    disable_stdout()  # (following call spews Google warning sometimes)
    info = spec.make_info()
    enable_stdout()
    assert info is not None
    scale = info["scales"][scale_index]
    resolution = scale["resolution"]
    start_coord = scale["voxel_offset"]
    size = scale["size"]
    end_coord = [a + b for (a, b) in zip(start_coord, size)]
    if index_resolution is None:
        index_resolution = resolution
    cvl = build_cv_layer(
        path=path,
        allow_slice_rounding=True,
        default_desired_resolution=index_resolution,
        index_resolution=index_resolution,
        data_resolution=resolution,
        interpolation_mode=info["type"],
    )
    bounds = VolumetricIndex.from_coords(start_coord, end_coord, index_resolution)
    return cvl, bounds


def disable_stdout():
    sys.stdout = open(os.devnull, "w")


def enable_stdout():
    sys.stdout = sys.__stdout__


# Stored state URL be like:
# https://spelunker.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/6704696916967424
# ID is the last part of this.

verify_cave_auth()

os.system("clear")
state_id = input_or_default("Neuroglancer state ID or URL", "6704696916967424")
if "/" in state_id:
    state_id = state_id.split("/")[-1]

state = client.state.get_state_json(state_id)  # ID from end of NG state URL
cell_layer_name = get_segmentation_layer_name(state, "cell segmentation")
cell_source = nglui.parser.get_layer(state, cell_layer_name)["source"]
print(f"Loading cell segmentation data from:\n{cell_source}")

other_layer_name = get_segmentation_layer_name(state, "other segmentation (for bounding box)")
other_source = nglui.parser.get_layer(state, other_layer_name)["source"]
volume_index = get_bounding_box(other_source)
print(f"Using bounds from:\n{other_source}")
print(f"Data bounds (nm): {volume_index.bbox}")

# Get the cell IDs in our region of interest
cell_cvl, cell_index = load_volume(cell_source)
resolution = [int(i) for i in cell_index.resolution]
print(f"Working resolution: {resolution}")

volume_index.allow_slice_rounding = True
volume_index.resolution = Vec3D(*resolution)
round_start = round(volume_index.start)
volume_index = VolumetricIndex.from_coords(
    round_start, round_start + volume_index.shape, volume_index.resolution
)
print(f"Volume index: {volume_index}")

cell_ids = cell_cvl[volume_index][0]

unique_ids = np.unique(cell_ids)
if unique_ids[0] == 0:
    unique_ids = unique_ids[1:]

print(f"This cutout contains {len(unique_ids)} cells.")

presyn_file = ""
presyn_ids: list[int] = []
if input("Enter presynaptic cell IDs [M]anually or from [File]? ").upper == "F":
    while True:
        presyn_file = input_or_default("File containing presynaptic cell IDs", "presyn_cells.txt")
        if presyn_file[-4:] == ".txt":
            presyn_ids = list(np.loadtxt(presyn_file, dtype=int))
        elif presyn_file[-4:] == ".csv":
            presyn_ids = list(
                np.genfromtxt(presyn_file, delimiter=",", skip_header=1, usecols=(0,), dtype=int)
            )
        else:
            print("Only .txt and .csv are supported")
            continue
        break

    print(f"{presyn_file} contains {len(presyn_ids)} IDs.")
else:
    presyn_file = "manual input"
    presyn_ids = [int(input("Cell ID: ").strip())]
    print("(Enter a blank ID to stop.)")
    while True:
        inp = input("Cell ID: ").strip()
        if not inp:
            break
        presyn_ids.append(int(inp))

if np.isin(presyn_ids, unique_ids).all():
    print("Verified that all these IDs are valid in this cutout.")
else:
    print(f"WARNING: Some entries in {presyn_file} are not found in this cutout.")

nowstr = datetime.now().strftime("%Y%m%d")  # new file daily (but same within each day)
out_path = input_or_default("Output layer GS path", f"gs://tmp_2w/joe/concact-{nowstr}")


def find_contact_mask(cell_ids, cell1, cell2, n=2):
    """
    Find and return a mask of all the places that cell1 and cell2 come within n*2
    steps of each other (in X and Y), within the cell_ids array.
    """
    # Create binary masks for cell1 and cell2
    mask_cell1 = cell_ids == cell1
    mask_cell2 = cell_ids == cell2

    # Define the structuring element for dilation (only dilate in X and Y)
    structuring_element = np.ones((n * 2 + 1, n * 2 + 1, 1))

    # Dilate the masks
    dilated_mask_cell1 = binary_dilation(mask_cell1, structure=structuring_element)
    dilated_mask_cell2 = binary_dilation(mask_cell2, structure=structuring_element)

    # Find the intersection of the dilated masks
    intersection_mask = dilated_mask_cell1 & dilated_mask_cell2
    return intersection_mask


def find_contact_clusters(cell_ids, cell1, cell2, n=2):
    """
    Find and return labeled clusters of contact areas where cell1 and cell2
    come within n*2 steps of each other (in X and Y), within the cell_ids array.
    """
    mask = find_contact_mask(cell_ids, cell1, cell2, n)
    cc_labels = cc3d.connected_components(mask, connectivity=26)
    return cc_labels


max_label = 0
neighbor_dist = 1
for cell_id in presyn_ids:
    print(f"Presynaptic cell ID: {cell_id}")

    aggregate_labels = np.zeros(cell_ids.shape, dtype=np.int32)
    count = 0
    max_count = len(unique_ids) - 1
    for other_id in unique_ids:
        if other_id == cell_id:
            continue
        count += 1
        cc_labels = find_contact_clusters(cell_ids, cell_id, other_id, neighbor_dist)
        contact_count = cc_labels.max()
        if contact_count == 0:
            continue
        contact_volume = (cc_labels > 0).sum()
        print(
            f"({count}/{max_count}) {cell_id} and {other_id} touch in {contact_count} place(s),"
            f" totalling {contact_volume} voxels"
        )
        if contact_volume / contact_count < 10:
            continue  # too few pixels to be real
        cc_labels[cc_labels > 0] += max_label
        aggregate_labels[cc_labels > 0] = cc_labels[cc_labels > 0]
        max_label = aggregate_labels.max()
        # if count > 3: break  # HACK!!!

# Now prepare a new segmentation volume for the output.
disable_stdout()
out_cvl = build_cv_layer(
    path=out_path,
    default_desired_resolution=resolution,
    index_resolution=resolution,
    data_resolution=resolution,
    allow_slice_rounding=True,
    interpolation_mode="segmentation",
    info_type="segmentation",
    info_data_type="int32",
    info_num_channels=1,
    info_overwrite=True,
    info_scales=[resolution],
    info_encoding="raw",
    info_chunk_size=[128, 128, 40],
    info_bbox=volume_index.bbox,
)
enable_stdout()
shape = (1, *volume_index.shape)
out_cvl[volume_index] = aggregate_labels.reshape(*shape)
print(f"Wrote {shape} data containing {max_label} contacts to: {out_path}")
