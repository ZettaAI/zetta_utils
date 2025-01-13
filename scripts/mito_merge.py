"""
This script takes a set of mitochondria, and outputs line annotations which can be applied
to merge the mitochondria into its parent cell.  In cases where a mito has *two* parent
cells (e.g., it has caused a split), this will also merge those parents together.
"""

"""
This script prototypes a process for finding contacts between known presynaptic
cells in a smallish cutout, and any other cells in that cutout.
"""
import csv
import os
import sys
from datetime import datetime
from math import floor

import cc3d
import google.auth
import nglui
import numpy as np
from caveclient import CAVEclient
from scipy.ndimage import binary_dilation

from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex, VolumetricLayer
from zetta_utils.layer.volumetric.cloudvol import build_cv_layer
from zetta_utils.layer.volumetric.precomputed import PrecomputedInfoSpec

client: CAVEclient
ng_state = None


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


def get_ng_state():
    global ng_state
    global client
    if ng_state == None:
        verify_cave_auth()
        # Stored state URL be like:
        # https://spelunker.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/6704696916967424
        # ID is the last part of this.
        state_id = input_or_default("Neuroglancer state ID or URL", "5312699228487680")
        if "/" in state_id:
            state_id = state_id.split("/")[-1]

        ng_state = client.state.get_state_json(state_id)  # ID from end of NG state URL
    return ng_state


def load_cell_seg_layer():
    """
    Prompt the user (if needed) for NG state and segmentation layer name;
    return the CloudVolume and VolumetricIndex for that layer.
    """
    state = get_ng_state()
    cell_layer_name = get_segmentation_layer_name(state, "segmentation")
    cell_source = nglui.parser.get_layer(state, cell_layer_name)["source"]
    print(f"Loading cell segmentation data from:\n{cell_source}")
    cell_cvl, cell_index = load_volume(cell_source)
    return cell_cvl, cell_index


def load_mito_points():
    """
    Prompt the user (if needed) for NG state and annotation layer name;
    then return a list of the Vec3D coordinates of points in that layer.
    """
    state = get_ng_state()
    points_layer_name = get_annotation_layer_name(state, "mito points")
    points_layer = nglui.parser.get_layer(state, points_layer_name)
    result = []
    for item in points_layer["annotations"]:
        result.append(round(Vec3D(*item["point"])))
    return result


def process_one(mito_point: Vec3D, seg_layer, resolution: Vec3D):
    """
    Process one mitochondrion, assumed to be the segment that contains
    the given point.  Add lines between this segment and any neighboring
    ones (i.e. across any valid neighbor contacts).
    """
    print(f"Processing mito at {list(mito_point)}...")
    # We need to define a bounding box that is reasonably small (for performance),
    # but guaranteed to contain the full mitochondrion (for correctness).
    # This should be safe (at 16, 16, 40):
    halfSizeInNm = Vec3D(4096, 4096, 4096)
    halfSizeInVox = round(halfSizeInNm / resolution)
    idx = VolumetricIndex.from_coords(
        mito_point - halfSizeInVox, mito_point + halfSizeInVox, resolution
    )
    chunk = seg_layer[idx][0]
    relative_point = floor(halfSizeInVox)
    mito_id = chunk[relative_point[0], relative_point[1], relative_point[2]]

    # Approach: create a binary mask for the mitochondrion, dilate this by a
    # voxel or two, and use that to mask the chunk.  See what segment IDs are
    # left; those are the ones the mito touches.
    mask = chunk == mito_id
    N = 1  # neighborhood size
    structuring_element = np.ones((N * 2 + 1, N * 2 + 1, 1))
    dilated_mask = binary_dilation(mask, structure=structuring_element)
    boundary_mask = dilated_mask & (~mask)
    boundary_area = np.sum(boundary_mask)
    print(f"   Mitochondrion segment ID: {mito_id}; surface area: {boundary_area}")
    neighbor_ids = chunk[boundary_mask]
    unique_ids, counts = np.unique(neighbor_ids, return_counts=True)
    for id, count in zip(unique_ids, counts):
        valid = "VALID" if count > boundary_area * 0.10 else "invalid"
        print(f"   Found contact ({count} voxels) with segment {id} ({valid})")


def main():
    os.system("clear")
    mito_points = load_mito_points()
    print(f"Loaded {len(mito_points)} mitochondria points")
    cell_cvl, cell_index = load_cell_seg_layer()

    resolution = round(cell_index.resolution)
    print(f"Working resolution: {list(resolution)}")

    process_one(mito_points[0], cell_cvl, resolution)

    sys.exit()

    presyn_file = ""
    presyn_ids: list[int] = []
    if input("Enter presynaptic cell IDs [M]anually or from [File]? ").upper == "F":
        while True:
            presyn_file = input_or_default(
                "File containing presynaptic cell IDs", "presyn_cells.txt"
            )
            if presyn_file[-4:] == ".txt":
                presyn_ids = list(np.loadtxt(presyn_file, dtype=int))
            elif presyn_file[-4:] == ".csv":
                presyn_ids = list(
                    np.genfromtxt(
                        presyn_file, delimiter=",", skip_header=1, usecols=(0,), dtype=int
                    )
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
        info_voxel_offset=volume_index.start,
        info_dataset_size=volume_index.shape,
        info_chunk_size=[128, 128, 40],
        on_info_exists="overwrite",
        info_add_scales_ref={
            "chunk_sizes": [128, 128, 40],
            "encoding": "raw",
            "resolution": resolution,
            "size": volume_index.shape,
            "voxel_offset": volume_index.start,
        },
        info_add_scales=[resolution],
        # info_add_scales_mode = 'replace',
        info_field_overrides={
            "type": "segmentation",
            "data_type": "int32",
            "num_channels": 1,
        },
    )
    enable_stdout()
    shape = (1, *volume_index.shape)
    out_cvl[volume_index] = aggregate_labels.reshape(*shape)
    print(f"Wrote {shape} data containing {max_label} contacts to: {out_path}")


if __name__ == "__main__":
    main()
