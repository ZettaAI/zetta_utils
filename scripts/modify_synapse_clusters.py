"""
This script takes a synapse segmentation volume, and a NG state with several layers of line annotations:
    "TP": True positives: these clusters will be copied to the output volume unchanged
    "FP": False positives: these clusters will be deleted
    "FN": False negatives: these clusters will be added to the output volume
This script then generates a new image volume with clusters as described above.
"""

import json
import sys
from typing import Any, Dict, List, Optional, Sequence, cast

import nglui.parser
import numpy as np
from caveclient import CAVEclient
from cloudvolume import CloudVolume
from google.cloud import storage
from neuroglancer.viewer_state import AxisAlignedBoundingBoxAnnotation
from scipy.ndimage import binary_dilation

from zetta_utils.geometry import BBox3D, Vec3D

cave_client: CAVEclient | None = None


def verify_cave_auth() -> bool:
    global cave_client

    cave_client = CAVEclient(datastack_name=None, server_address="https://global.daf-apis.com")

    try:
        cave_client.state
        return True  # no exception?  All's good!
    except:
        pass
    print("Authentication needed.")
    print("Go to: https://global.daf-apis.com/auth/api/v1/create_token")
    token = input("Enter token: ")
    cave_client.auth.save_token(token=token, write_to_server_file=True)
    return True


def input_or_default(prompt: str, value: str) -> str:
    response = input(f"{prompt} [{value}]: ")
    if response == "":
        response = value
    return response


def get_layer_name_of_type(state, layer_type: str, prompt: str) -> str:
    names = nglui.parser.layer_names(state)
    numToName = {}
    for i, name in enumerate(names, start=1):
        layer = nglui.parser.get_layer(state, name)
        # print(f"{i}: {name} ({layer['type']})")
        if layer["type"] == layer_type:
            numToName[i] = name
    if len(numToName) == 0:
        print(f"No {layer_type} layers found in this state.")
        sys.exit()
    elif len(numToName) == 1:
        _, name = numToName.popitem()
        print(f"[{name}]")
        return name
    while True:
        for i, name in numToName.items():
            print(f"{i}. {name}")
        choice = input(f"Enter layer name or number ({prompt}): ")
        if choice in names:
            return choice
        ichoice = int(choice)
        if ichoice in numToName:
            return numToName[ichoice]


def get_annotation_layer_name(state, prompt: str):
    return get_layer_name_of_type(state, "annotation", prompt)


def get_segmentation_layer_name(state, prompt: str):
    return get_layer_name_of_type(state, "segmentation", prompt)


def dimToNm(valAndUnit: List) -> float:
    """
    Parse a dimension as found in NG and precomputed states, such as
    [16, "nm"] or [1.6E-8, "m"], and return it as simply a number of nm
    (e.g. 16 for both of the above examples).
    """
    value, unit = valAndUnit
    if unit == "m":
        value *= 1e9
    elif unit == "mm":
        value *= 1e6
    elif unit == "um" or unit == "µm":
        value *= 1e3
    elif unit != "nm":
        raise ValueError(f"Unknown unit in dimension: {valAndUnit}")
    return value


def read_gcs_json(bucket_path):
    # Parse bucket and blob path
    bucket_name = bucket_path.split("/")[2]
    blob_path = "/".join(bucket_path.split("/")[3:])

    # Initialize client
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    # Download and parse JSON
    json_str = blob.download_as_text()
    data = json.loads(json_str)
    return data


def get_resolution(ng_layer_data) -> Vec3D:
    """
    Returns the resolution in [x,y,z] nm of a Neuroglancer layer.
    """
    if "voxelSize" in ng_layer_data:
        return Vec3D(*ng_layer_data["voxelSize"])
    source = ng_layer_data["source"]
    if isinstance(source, str):
        json = read_gcs_json(source + "/info")
        if "scales" in json:
            scale0 = json["scales"][0]
            result: Vec3D = Vec3D(*scale0["resolution"])
            return result
        dims = json["dimensions"]
        result = Vec3D(dimToNm(dims["x"]), dimToNm(dims["y"]), dimToNm(dims["z"]))
        return result
    transform = source["transform"]
    if "inputDimensions" in transform:
        dims = transform["inputDimensions"]
    else:
        # No input dimensions defined?  Assume output equals input.
        dims = transform["outputDimensions"]
    if "0" in dims:
        result = Vec3D(dimToNm(dims["0"]), dimToNm(dims["1"]), dimToNm(dims["2"]))
    else:
        result = Vec3D(dimToNm(dims["x"]), dimToNm(dims["y"]), dimToNm(dims["z"]))
    return result


def get_bounding_boxes(state, layer_name: str, return_union: bool = False):
    """
    Returns a list of BBox3D objects representing the bounding boxes found in the
    given layer of a Neuroglancer state.  These boxes correctly take into account
    the layer resolution (dimensions), and we also ensure that each box is properly
    ordered with the smallest coordinate first and the largest coordinate last.

    Optionally, returns the union of all the bounding boxes as a second return value.
    """
    layer_data = nglui.parser.get_layer(state, layer_name)
    resolution = get_resolution(layer_data)
    print(f"Resolution of bounding boxes: {resolution}")
    bboxes = []
    union_bbox = None
    for item in layer_data["annotations"]:
        if item["type"] != "axis_aligned_bounding_box":
            continue
        ng_bbox = AxisAlignedBoundingBoxAnnotation(pointA=item["pointA"], pointB=item["pointB"])
        bbox = BBox3D.from_ng_bbox(ng_bbox, resolution)
        bboxes.append(bbox)
        print(f"Found bounding box: {bbox.pformat(resolution)}")
        if union_bbox is None:
            union_bbox = bbox
        else:
            union_bbox = union_bbox.supremum(bbox)  # type: ignore[unreachable]
    if return_union:
        return bboxes, union_bbox
    return bboxes


def get_annotation_lines(state, layer_name: str, resolution: Sequence[int]):
    """
    Returns a list of line annotations from a Neuroglancer state, with endpoints
    converted to integer Vec3Ds at the given resolution.
    """
    layer_data = nglui.parser.get_layer(state, layer_name)
    layer_res = get_resolution(layer_data)
    desired_res = Vec3D(*resolution)
    result = layer_data["annotations"]
    for i in range(len(result) - 1, -1, -1):
        if result[i]["type"] != "line":
            del result[i]
            continue
        pointA: Vec3D = Vec3D(*result[i]["pointA"])
        pointB: Vec3D = Vec3D(*result[i]["pointB"])
        if layer_res == desired_res:
            result[i]["pointA"] = pointA.int()
            result[i]["pointB"] = pointB.int()
        else:
            result[i]["pointA"] = (pointA * layer_res / desired_res).int()
            result[i]["pointB"] = (pointB * layer_res / desired_res).int()
    return result


def get_labels_along_line(volume, start_point: Vec3D, end_point: Vec3D):
    """Return set of nonzero values in volume along the line from start to end"""
    labels = set()
    x0, y0, z0 = round(start_point)
    x1, y1, z1 = round(end_point)
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dz = abs(z1 - z0)

    xs = 1 if x1 > x0 else -1
    ys = 1 if y1 > y0 else -1
    zs = 1 if z1 > z0 else -1

    # Driving axis is the one with max delta
    if dx >= dy and dx >= dz:
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while x0 != x1:
            if volume[x0, y0, z0, 0] != 0:
                labels.add(int(volume[x0, y0, z0, 0]))
            x0 += xs
            if p1 >= 0:
                y0 += ys
                p1 -= 2 * dx
            if p2 >= 0:
                z0 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz

    elif dy >= dx and dy >= dz:
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while y0 != y1:
            if volume[x0, y0, z0, 0] != 0:
                labels.add(int(volume[x0, y0, z0, 0]))
            y0 += ys
            if p1 >= 0:
                x0 += xs
                p1 -= 2 * dy
            if p2 >= 0:
                z0 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz

    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while z0 != z1:
            if volume[x0, y0, z0, 0] != 0:
                labels.add(int(volume[x0, y0, z0, 0]))
            z0 += zs
            if p1 >= 0:
                y0 += ys
                p1 -= 2 * dz
            if p2 >= 0:
                x0 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx

    return labels


def erase_clusters_along_line(binary_mask, labels, bbox: BBox3D, line: Dict, resolution: Vec3D):
    """Erase clusters along a line in a binary mask.  Return True if any clusters were erased,
    False otherwise."""
    pointA = line["pointA"]
    pointB = line["pointB"]
    if not bbox.line_intersects(pointA, pointB, resolution):
        return False

    # Convert line endpoints to bbox-relative coordinates
    s = (bbox.start / resolution).int()
    start = pointA - s
    end = pointB - s

    # Find labels that intersect with the line.  We'll try up to three different Z positions,
    # since that can often make the difference between finding the clustor or not.
    for delta in [0, 1, -1]:
        dvec = Vec3D(0, 0, delta)
        intersecting_labels = get_labels_along_line(labels, start + dvec, end + dvec)
        if len(intersecting_labels) > 0:
            break
    if len(intersecting_labels) == 0:
        print(f"No labels found for line from {tuple(pointA)} to {tuple(pointB)}")
        return False
    elif len(intersecting_labels) > 1:
        print(
            f"Multiple labels found for line from {tuple(pointA)} to {tuple(pointB)}: {str(intersecting_labels)}"
        )

    # Clear those components from binary mask
    for label in intersecting_labels:
        binary_mask[labels == label] = 0
    return True


def add_cluster_along_line(binary_mask, cell_seg, bbox: BBox3D, line: Dict, resolution: Vec3D):
    """Add a cluster along a line in a binary mask.  The line identifies two cells
    via cell_seg; we dilate those a bit and find the overlap to determine the contact
    region, and then fill in the mask from the overlap within some radius of the line."""
    pointA = line["pointA"]
    pointB = line["pointB"]
    if not bbox.line_intersects(pointA, pointB, resolution):
        return

    # Convert line endpoints to bbox-relative coordinates
    s = (bbox.start / resolution).int()
    start = pointA - s
    end = pointB - s

    # Get the cell ID for each endpoint
    cell_idA = cell_seg[start[0], start[1], start[2], 0]
    cell_idB = cell_seg[end[0], end[1], end[2], 0]

    # Create and dilate a binary mask for each cell ID
    cell_maskA = (cell_seg == cell_idA).astype(np.uint8)
    cell_maskB = (cell_seg == cell_idB).astype(np.uint8)
    cell_maskA = binary_dilation(cell_maskA, np.ones((3, 3, 2, 1)))
    cell_maskB = binary_dilation(cell_maskB, np.ones((3, 3, 2, 1)))

    # Find the overlap between the two masks
    overlap = np.logical_and(cell_maskA, cell_maskB)
    if not np.any(overlap):
        print(
            f"No overlap found between {tuple(pointA)} and {tuple(pointB)} (cells {cell_idA} and {cell_idB})"
        )
        return

    # Create a round nonisotropic mask around the line midpoint
    midpoint = ((start + end) / 2).int()
    xy_radius = 10
    z_radius = 2
    x, y, z = np.ogrid[
        -midpoint[0] : binary_mask.shape[0] - midpoint[0],
        -midpoint[1] : binary_mask.shape[1] - midpoint[1],
        -midpoint[2] : binary_mask.shape[2] - midpoint[2],
    ]
    radius_mask = ((x * x + y * y) <= xy_radius * xy_radius) & (np.abs(z) <= z_radius)
    radius_mask = radius_mask[..., np.newaxis]  # (make 4-dimensional)

    # Update the binary mask with the overlap, but only within the radius mask
    binary_mask[np.logical_and(overlap, radius_mask)] = 255


def add_terminal(
    cell_seg, bbox: BBox3D, line: Dict, which_point: str, resolution: Vec3D, output_cv: CloudVolume
):
    """Add a terminal cluster for one end of a line in a binary mask."""
    point = line["point" + which_point]
    if not bbox.contains(point, resolution):
        return

    # Convert endpoint to bbox-relative coordinates, and find the cell ID
    s = (bbox.start / resolution).int()
    local_point = point - s
    cell_id = cell_seg[local_point[0], local_point[1], local_point[2], 0]
    cell_mask = (cell_seg == cell_id).astype(np.uint8)

    # Create a round nonisotropic mask around the endpoint
    xy_radius = 10
    z_radius = 2
    x, y, z = np.ogrid[
        -local_point[0] : cell_seg.shape[0] - local_point[0],
        -local_point[1] : cell_seg.shape[1] - local_point[1],
        -local_point[2] : cell_seg.shape[2] - local_point[2],
    ]
    radius_mask = ((x * x + y * y) <= xy_radius * xy_radius) & (np.abs(z) <= z_radius)
    radius_mask = radius_mask[..., np.newaxis]  # (make 4-dimensional)

    # Update the binary mask with the cell mask, but only within the radius mask
    output_cv[np.logical_and(cell_mask, radius_mask)] = 255


def process_bbox(
    source_cv: CloudVolume,
    cell_cv: CloudVolume,
    bbox: BBox3D,
    lines: dict,
    output_cv: CloudVolume,
    presyn_output_cv: CloudVolume,
    postsyn_output_cv: CloudVolume,
):
    """
    Process a bounding box by erasing any clusters that intersect with the FP lines,
    adding new clusters for each FN line, and leaving the TP lines unchanged.
    Note that line endpoints must be in the same resolution as source_cv.
    """
    assert (source_cv.resolution == output_cv.resolution).all()

    # Pad the bounding box a bit, to ensure that synapses fit entirely within
    resolution: Vec3D = Vec3D(*source_cv.resolution)
    bbox = bbox.padded(pad=[20, 20, 10], resolution=resolution)

    # Get the segmentation data from the source volume
    s = round(bbox.start / resolution)
    e = round(bbox.end / resolution)
    syn_data = source_cv[s[0] : e[0], s[1] : e[1], s[2] : e[2]]
    cell_data = cell_cv[s[0] : e[0], s[1] : e[1], s[2] : e[2]]

    # Convert synapse segmentation data to binary mask (255 where nonzero, 0 where zero)
    binary_mask = (syn_data > 0).astype(np.uint8) * 255

    # Handle FP lines by erasing any cluster these cross
    for line in lines["FP"]:
        erase_clusters_along_line(binary_mask, syn_data, bbox, line, resolution)

    # Handle FN lines by adding new clusters for each line
    for line in lines["FN"]:
        add_cluster_along_line(binary_mask, cell_data, bbox, line, resolution)

    # Write to file
    output_cv[s[0] : e[0], s[1] : e[1], s[2] : e[2]] = binary_mask

    # If presynaptic output is desired, generate presynaptic clusters for TP and FN lines.
    if presyn_output_cv is not None:
        presyn_mask = np.zeros(binary_mask.shape, dtype=np.uint8)
        for line in lines["TP"]:
            add_terminal(cell_data, bbox, line, "A", resolution, presyn_mask)
        for line in lines["FN"]:
            add_terminal(cell_data, bbox, line, "A", resolution, presyn_mask)
        presyn_output_cv[s[0] : e[0], s[1] : e[1], s[2] : e[2]] = presyn_mask

    if postsyn_output_cv is not None:
        postsyn_mask = np.zeros(binary_mask.shape, dtype=np.uint8)
        for line in lines["TP"]:
            add_terminal(cell_data, bbox, line, "B", resolution, postsyn_mask)
        for line in lines["FN"]:
            add_terminal(cell_data, bbox, line, "B", resolution, postsyn_mask)
        postsyn_output_cv[s[0] : e[0], s[1] : e[1], s[2] : e[2]] = postsyn_mask


def create_volume(source_cv: CloudVolume, output_path: str) -> CloudVolume:
    """
    Create a new CloudVolume instance based on the source volume's parameters.

    :param source_cv: Source CloudVolume to copy parameters from
    :param output_path: Path where the new volume will be created

    :return: The newly created volume
    """
    if not output_path.startswith("precomputed://"):
        output_path = "precomputed://" + output_path

    info = CloudVolume.create_new_info(
        num_channels=1,
        layer_type="image",
        data_type="uint8",
        encoding="raw",
        resolution=source_cv.resolution,
        voxel_offset=source_cv.voxel_offset,
        mesh="mesh",
        chunk_size=source_cv.chunk_size,
        volume_size=source_cv.volume_size,
    )
    output_cv = CloudVolume(output_path, info=info)
    output_cv.commit_info()
    output_cv.non_aligned_writes = True
    output_cv.fill_missing = True
    output_cv.delete_black_uploads = True

    return output_cv


# def main():
if __name__ == "__main__":
    verify_cave_auth()
    assert cave_client is not None
    print(f"Enter Neuroglancer state link or ID:")
    source_path = input("> ")
    state_id = source_path.split("/")[-1]  # in case full URL was given
    state = cave_client.state.get_state_json(state_id)

    # Get the synapse segmentation layer
    seg_layer_name = get_segmentation_layer_name(state, "Synapse Segmentation")
    layer_data = nglui.parser.get_layer(state, seg_layer_name)
    source_path = layer_data["source"]
    if "/|neuroglancer-precomputed:" in source_path:
        source_path = source_path.split("/|neuroglancer-precomputed:")[0]
    print(f"Synapse segmentation path: {source_path}")
    source_cv = CloudVolume(source_path)
    source_res = [int(i) for i in source_cv.resolution]
    print(f"   Source resolution: {source_res}")
    print(f"   Source offset:     {source_cv.voxel_offset}")
    print(f"   Source data size:  {source_cv.volume_size}")

    # Get the cell segmentation layer (likely to be a Graphene layer)
    cell_layer_name = get_segmentation_layer_name(state, "Cell Segmentation")
    cell_layer_data = nglui.parser.get_layer(state, cell_layer_name)
    cell_source_path = cell_layer_data["source"]
    if isinstance(cell_source_path, dict):
        cell_source_path = cell_source_path["url"]
    print(f"Cell segmentation path: {cell_source_path}")
    cell_cv = CloudVolume(cell_source_path)
    cell_cv.agglomerate = True  # get root IDs, not supervoxel IDs
    cell_res = [int(i) for i in cell_cv.resolution]
    print(f"   Cell resolution: {cell_res}")
    assert (cell_cv.resolution == source_cv.resolution).all()

    # Get the bounding box(es) which define the work space
    bbox_layer_name = get_annotation_layer_name(state, "Bounding Boxes")
    bboxes, union_bbox = get_bounding_boxes(state, bbox_layer_name, return_union=True)
    print(f"Found {len(bboxes)} bounding boxes")
    print(f"Union of all bounding boxes: {union_bbox.pformat(tuple(source_res))}")

    # Get the FP, TP, and FN layers
    layer_names = nglui.parser.layer_names(state)
    fp_layer_name = (
        "FP" if "FP" in layer_names else get_annotation_layer_name(state, "False Positives")
    )
    tp_layer_name = (
        "TP" if "TP" in layer_names else get_annotation_layer_name(state, "True Positives")
    )
    fn_layer_name = (
        "FN" if "FN" in layer_names else get_annotation_layer_name(state, "False Negatives")
    )
    lines = {}
    lines["FP"] = get_annotation_lines(state, fp_layer_name, source_res)
    lines["TP"] = get_annotation_lines(state, tp_layer_name, source_res)
    lines["FN"] = get_annotation_lines(state, fn_layer_name, source_res)

    # Initialize the output volumes
    cleft_output_path = input_or_default(
        "Enter output path for CLEFT DETECTION", "gs://tmp_2w/joe/syntest_cleft"
    )
    cleft_output_cv = create_volume(source_cv, cleft_output_path)

    print("For the following optional paths, enter '-' to skip.")
    presyn_output_path = input_or_default(
        "(Optional) output path for PRESYNAPTIC TERMINALS", "gs://tmp_2w/joe/syntest_presyn"
    )
    presyn_output_cv = (
        create_volume(source_cv, presyn_output_path) if presyn_output_path != "-" else None
    )

    postsyn_output_path = input_or_default(
        "(Optional) output path for POSTSYNAPTIC TERMINALS", "gs://tmp_2w/joe/syntest_postsyn"
    )
    postsyn_output_cv = (
        create_volume(source_cv, postsyn_output_path) if postsyn_output_path != "-" else None
    )

    # Now, iterate over and process each bounding box
    for bbox in bboxes:
        print(f"Processing bounding box: {bbox.pformat(tuple(source_res))}")
        data = process_bbox(
            source_cv, cell_cv, bbox, lines, cleft_output_cv, presyn_output_cv, postsyn_output_cv
        )
    print("Done!")

# if __name__ == "__main__":
#     main()
