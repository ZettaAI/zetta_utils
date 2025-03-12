"""
This script takes a segmentation volume, and a NG state with several layers of line annotations:
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
    offset = Vec3D(0, 0, -0.5)  # accounts for how NG displays annotations off by 0.5 voxel in Z
    for i in range(len(result) - 1, -1, -1):
        if result[i]["type"] != "line":
            del result[i]
            continue
        pointA = Vec3D(*result[i]["pointA"]) + offset
        pointB = Vec3D(*result[i]["pointB"]) + offset
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


def process_bbox(source_cv: CloudVolume, bbox: BBox3D, lines: dict, output_cv: CloudVolume):
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
    data = source_cv[s[0] : e[0], s[1] : e[1], s[2] : e[2]]

    # Convert segmentation data to binary mask (255 where nonzero, 0 where zero)
    binary_mask = (data > 0).astype(np.uint8) * 255

    # Handle FP lines by erasing any cluster these cross
    for line in lines["FP"]:
        pointA = line["pointA"]
        pointB = line["pointB"]
        if not bbox.line_intersects(pointA, pointB, resolution):
            continue

        # Convert line endpoints to bbox-relative coordinates
        start = pointA - s
        end = pointB - s

        # Find labels that intersect with the line
        intersecting_labels = get_labels_along_line(data, start, end)
        print(
            f"Clearing labels for line from {tuple(pointA)} to {tuple(pointB)}: {str(intersecting_labels)}"
        )

        # Clear those components from binary mask
        for label in intersecting_labels:
            binary_mask[data == label] = 0

    # Write to file
    output_cv[s[0] : e[0], s[1] : e[1], s[2] : e[2]] = binary_mask


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
    print(f"Synapse segmentation path: {source_path}")
    source_cv = CloudVolume(source_path)
    source_res = [int(i) for i in source_cv.resolution]
    print(f"   Source resolution: {source_res}")
    print(f"   Source offset:     {source_cv.voxel_offset}")
    print(f"   Source data size:  {source_cv.volume_size}")

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

    # Initialize the output volume
    output_path = input_or_default("Enter output path", "gs://tmp_2w/joe/syntest")
    if not output_path.startswith("precomputed://"):
        output_path = "precomputed://" + output_path
    info = CloudVolume.create_new_info(
        num_channels=1,
        layer_type="image",
        data_type="uint8",
        encoding="raw",
        resolution=source_res,
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

    # Now, iterate over and process each bounding box
    for bbox in bboxes:
        print(f"Processing bounding box: {bbox.pformat(tuple(source_res))}")
        data = process_bbox(source_cv, bbox, lines, output_cv)
    print("Done!")

# if __name__ == "__main__":
#     main()
