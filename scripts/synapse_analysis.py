"""
This script compares a set of synapse lines to another (ground truth) set.
"""
import csv
import io
import json
import readline
import sys
from math import ceil, floor
from typing import Any, Dict, List, Optional, Sequence, cast

import nglui
import numpy as np
from caveclient import CAVEclient
from cloudfiles import CloudFile
from cloudvolume import CloudVolume
from google.cloud import storage
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex
from zetta_utils.layer.volumetric.annotation import build_annotation_layer
from zetta_utils.layer.volumetric.cloudvol import build_cv_layer

client: Optional[CAVEclient] = None
source_path = ""


def verify_cave_auth():
    # pylint: disable-next=global-statement
    global client
    client = CAVEclient()
    try:
        # pylint: disable-next=pointless-statement    # shut up, pylint
        client.state
        return  # no exception?  All's good!
    # pylint: disable-next=bare-except
    except:
        pass
    print("Authentication needed.")
    print("Go to: https://global.daf-apis.com/auth/api/v1/create_token")
    token = input("Enter token: ")
    client.auth.save_token(token=token)


def input_or_default(prompt, value):
    response = input(f"{prompt} [{value}]: ")
    if response == "":
        response = value
    return response


def input_vec3D(prompt="", default=None) -> Optional[Vec3D]:
    while True:
        s = input(prompt + (f" [{default.x}, {default.y}, {default.z}]" if default else "") + ": ")
        if s == "" and default:
            return default
        try:
            x, y, z = map(float, s.replace(",", " ").strip(" ()").split())
            return Vec3D(x, y, z)
        # pylint: disable-next=bare-except
        except:
            print("Enter x, y, and z values separated by commas or spaces.")


def input_vec3Di(prompt="", default=None) -> Optional[Vec3D]:
    v = input_vec3D(prompt, default)
    if v is None:
        return None
    return round(v)


def get_layer_name_of_type(state, layer_type: str) -> str:
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
        choice = input("Enter layer name or number: ")
        if choice in names:
            return choice
        ichoice = int(choice)
        if ichoice in numToName:
            return numToName[ichoice]


def get_annotation_layer_name(state):
    return get_layer_name_of_type(state, "annotation")


def get_segmentation_layer_name(state):
    return get_layer_name_of_type(state, "segmentation")


def read_gcs_json(bucket_path):
    # Parse bucket and blob path
    bucket_path = bucket_path.removeprefix("precomputed://")
    bucket_name = bucket_path.split("/")[2]
    blob_path = "/".join(bucket_path.split("/")[3:])
    print("bucket_path: " + bucket_path)
    print("bucket_name: " + bucket_name)

    # Initialize client
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    # Download and parse JSON
    json_str = blob.download_as_text()
    data = json.loads(json_str)
    return data


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


def get_resolution(ng_layer_data) -> Vec3D:
    if "voxelSize" in ng_layer_data:
        return Vec3D(*ng_layer_data["voxelSize"])
    source = ng_layer_data["source"]
    if isinstance(source, str):
        if "/|neuroglancer-precomputed:" in source:
            source = source.split("/|neuroglancer-precomputed:")[0]
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


def read_csv(source_path):
    cf = CloudFile(source_path)
    data = cf.get()
    lines: list[dict] = []
    reader = csv.DictReader(io.StringIO(data.decode("utf-8")))
    for row in reader:
        lines.append(
            {
                "id": str(len(lines) + 1),
                "type": "line",
                "pointA": [int(row["presyn_x"]), int(row["presyn_y"]), int(row["presyn_z"])],
                "pointB": [int(row["postsyn_x"]), int(row["postsyn_y"]), int(row["postsyn_z"])],
            }
        )
    return lines


def load_annotations(forPurpose: str, default_box: Optional[dict] = None):
    global source_path
    layer = None
    lineItems = None
    boxItem: Optional[dict] = None
    resolution = None
    print(f"Enter Neuroglancer state link or ID, or a GS file path for {forPurpose}:")
    inp = input("> ")
    if inp:
        source_path = inp
    if "/|neuroglancer-precomputed:" in source_path:
        source_path = source_path.split("/|neuroglancer-precomputed:")[0]
    if source_path.startswith("gs:"):
        if source_path.endswith(".df") or source_path.endswith(".csv"):
            annotations = read_csv(source_path)
            resolution = input_vec3Di("Resolution")
            return annotations, resolution, None
        layer = build_annotation_layer(source_path, mode="read")
        resolution = layer.backend.index.resolution
    else:
        verify_cave_auth()
        state_id = source_path.split("/")[-1]  # in case full URL was given

        assert client is not None
        print(f"Loading state {state_id}...")
        state = client.state.get_state_json(state_id)
        print(f"Select annotation layer containing synapses to import for {forPurpose}:")
        anno_layer_name = get_annotation_layer_name(state)
        data = nglui.parser.get_layer(state, anno_layer_name)
        resolution = get_resolution(data)

        if "annotations" in data:
            lineItems = [item for item in data["annotations"] if item["type"] == "line"]
            boxItem = next(
                (
                    item
                    for item in data["annotations"]
                    if item["type"] == "axis_aligned_bounding_box"
                ),
                None,
            )
            if boxItem != None:
                boxItem = cast(Dict[Any, Any], boxItem)  # stupid mypy
                boxItem["resolution"] = list(resolution)
        elif "source" in data:
            print("Precomputed annotation layer.")
            source = data["source"]
            if "/|neuroglancer-precomputed:" in source:
                source = source.split("/|neuroglancer-precomputed:")[0]
            layer = build_annotation_layer(source, mode="read")
        else:
            print("Neither 'annotations' nor 'source' found in layer data.  I'm stumped.")
            sys.exit()
    if lineItems is None and layer is not None:
        if default_box is not None:
            bbox_start: Optional[Vec3D] = Vec3D(*default_box["pointA"])
            assert bbox_start is not None
            bbox_end: Optional[Vec3D] = Vec3D(*default_box["pointB"])
            assert bbox_end is not None
            box_res: Optional[Vec3D] = Vec3D(*default_box["resolution"])
            assert box_res is not None
            # Using from_points here so as to straighten out a bbox that might
            # be defined backwards on some dimensions.
            bbox = BBox3D.from_points([bbox_start, bbox_end], box_res, epsilon=0)
            print(f"Reading lines within {bbox}...")
            lines = layer.backend.read_in_bounds(bbox, strict=True)
            boxItem = default_box
        else:
            opt = ""
            while opt not in ("A", "B"):
                opt = input("Read [A]ll lines, or only within some [B]ounds? ").upper()
            if opt == "B":
                bbox_start = input_vec3Di("  Bounds start")
                assert bbox_start is not None
                bbox_end = input_vec3Di("    Bounds end")
                assert bbox_end is not None
                resolution = input_vec3Di("    Resolution")
                assert resolution is not None
                bbox = BBox3D.from_coords(bbox_start, bbox_end, resolution)
                lines = layer.backend.read_in_bounds(bbox, strict=True)
                boxItem = {
                    "pointA": list(bbox_start),
                    "pointB": list(bbox_end),
                    "resolution": list(resolution),
                    "type": "axis_aligned_bounding_box",
                }
            else:
                lines = layer.backend.read_all()
        lineItems = [
            {"id": hex(l.id)[2:], "type": "line", "pointA": l.start, "pointB": l.end}
            for l in lines
        ]
    return lineItems, resolution, boxItem


def load_segmentation():
    global source_path
    layer = None
    lineItems = None
    boxItem = None
    resolution = None
    print(f"Enter Neuroglancer state link or ID, or a GS file path for SEGMENTATION:")
    inp = input("> ")
    if inp:
        source_path = inp
    if source_path.startswith("gs:"):
        resolution = input_vec3Di("Resolution")
    else:
        verify_cave_auth()
        state_id = source_path.split("/")[-1]  # in case full URL was given

        assert client is not None
        state = client.state.get_state_json(state_id)
        print(f"Select segmentation layer to use for identifying cells:")
        seg_layer_name = get_segmentation_layer_name(state)
        layer_data = nglui.parser.get_layer(state, seg_layer_name)
        source_path = layer_data["source"]
        if isinstance(source_path, dict):  # type: ignore[unreachable]
            source_path = source_path["url"]  # type: ignore[unreachable]
        # resolution = get_resolution(layer_data)
    if "/|neuroglancer-precomputed:" in source_path:
        source_path = source_path.split("/|neuroglancer-precomputed:")[0]
    print(f"Loading segmentation from {source_path}...")
    seg_cvl = CloudVolume(source_path)
    seg_cvl.agglomerate = True  # get root IDs, not supervoxel IDs
    return seg_cvl, Vec3D(*seg_cvl.resolution)
    # seg_cvl = build_cv_layer(
    #     path=source_path,
    #     allow_slice_rounding=True,
    #     index_resolution=resolution,
    #     data_resolution=resolution,
    #     interpolation_mode="nearest",
    #     readonly=True,
    # )
    # return seg_cvl, resolution


def analyze_points(points_A, points_B, valid_test=None) -> dict:
    """
    Match up predicted points to ground-truth points, allowing matches only
    for pairs of points that pass valid_test (if given).  Return stats dict.

    :param points_A: sequence of Vec3Ds representing ground truth
    :param points_B: sequence of Vec3Ds representing predictions
    """
    result: Dict[str, Any] = {}
    result["count_A"] = len(points_A)
    result["count_B"] = len(points_B)

    max_distance = 1000  # was: 250  # (nm)

    # Build KD-tree for set B; this lets us efficiently query for the point in B closest
    # to any other point (e.g., points in A), or even get ALL the distances to points in
    # B of a given point in A.
    kdtree = cKDTree(points_B)

    # Compute a distance matrix
    distance_matrix = np.full((len(points_A), len(points_B)), np.inf)  # Initialize with inf values
    for i, point in enumerate(points_A):
        # Get distances to this point, up to max_distance, filling in inf for remaining slots
        distances, indices = kdtree.query(
            point, k=len(points_B), distance_upper_bound=max_distance
        )
        if isinstance(distances, float):
            distances = [distances]
            indices = [indices]
        for dist, j in zip(distances, indices):
            if dist < max_distance and (
                valid_test is None or valid_test(points_A[i], points_B[j])
            ):
                distance_matrix[i, j] = dist  # Only store distance if valid

    # Replace np.inf with a large finite value (to make the cost matrix solvable)
    large_value = 1e9  # Arbitrarily large value to discourage infeasible matches
    distance_matrix[np.isinf(distance_matrix)] = large_value

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    # Filter matches by max distance
    matches = [(i, j) for i, j in zip(row_ind, col_ind) if distance_matrix[i, j] < max_distance]

    # Now, gather stats, considering every match to indicate a true positive.
    tp = len(matches)
    fp = len(points_B) - tp
    fn = len(points_A) - tp
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    result["true_pos"] = tp
    result["false_pos"] = fp
    result["false_neg"] = fn
    result["precision"] = precision
    result["recall"] = recall
    if precision + recall > 0:
        result["f1"] = 2 * precision * recall / (precision + recall)
    else:
        result["f1"] = "(NaN)"

    distances = [distance_matrix[m[0], m[1]] for m in matches]
    distances = sorted(distances)
    if len(distances) > 0:
        result["mean_dist"] = sum(distances) / len(distances)
        result["median_dist"] = distances[len(distances) // 2]
        result["max_dist"] = distances[-1]
        result["min_dist"] = distances[0]
    else:
        result["mean_dist"] = "-"
        result["median_dist"] = "-"
        result["max_dist"] = "-"
        result["min_dist"] = "-"

    # Detailed match info, for debugging
    result["matches_list"] = matches
    result["tp_points_A"] = [points_A[i] for i, _ in matches]
    result["tp_points_B"] = [points_B[i] for _, i in matches]
    result["fp_points_B"] = list([p for p in points_B if p not in result["tp_points_B"]])
    result["fn_points_A"] = list([p for p in points_A if p not in result["tp_points_A"]])

    return result


def print_stats(stats: dict, title: str):
    print()
    print(title)
    print("-" * len(title))
    print(f"True Positives:  {stats['true_pos']}")
    print(f"False Positives: {stats['false_pos']}")
    print(f"False Negatives: {stats['false_neg']}")
    print(f"Precision:       {round(stats['precision'], 3)}")
    print(f"Recall:          {round(stats['recall'], 3)}")
    if isinstance(stats["f1"], float):
        print(f"F1 Score:        {round(stats['f1'], 3)}")
        print(f"Distance Range:  {round(stats['min_dist'])} - {round(stats['max_dist'])}")
        print(f"Mean Distance:   {round(stats['mean_dist'])}")
        print(f"Median Distance: {round(stats['median_dist'])}")


def get_points(items: Sequence[dict], key: str, resolution: Vec3D) -> List[Vec3D]:
    """
    Get points out of a list of annotation items, converted to Vec3D in nm.
    """
    result = []
    for item in items:
        if key == "center" and key not in item:
            p: Vec3D = (Vec3D(*item["pointA"]) + Vec3D(*item["pointB"])) / 2
        else:
            p = Vec3D(*item[key])
        result.append(p * resolution)
    return result


def lookup_seg_id(point, seg_data, idx):
    """
    Look up the segmentation ID for a given point (Vec3D in nm).
    """
    relative_point = floor(point / idx.resolution - idx.start)
    try:
        return seg_data[relative_point[0], relative_point[1], relative_point[2]]
    except:
        return None


def lookup_seg_ids(points, seg_data, idx, result_map):
    """
    Look up each point in seg_data, and add the mapping from point
    to id to result_map.  If the point is out of bounds (of idx),
    then instead REMOVE IT from the points list.
    """
    for i in range(len(points) - 1, -1, -1):
        id = lookup_seg_id(points[i], seg_data, idx)
        if id is None:
            del points[i]
        else:
            result_map[points[i]] = id


def print_as_annotations(points, items, key, resolution):
    result = []
    for p in points:
        matching_items = [item for item in items if item[key] * resolution == p]
        if len(matching_items) == 1:
            m = matching_items[0]
            pointA = list(m["pointA"])
            pointB = list(m["pointB"])
            result.append(
                "{"
                + f""""type": "line", "id": "{m['id']}", "pointA": {pointA}, "pointB": {pointB}"""
                + "}"
            )
        else:
            print(f"{p}: {matching_items}")

    print(",\n".join(result))


def line_in_bounds(line, line_res, idx):
    a = Vec3D(*line["pointA"]) * line_res / idx.resolution
    b = Vec3D(*line["pointB"]) * line_res / idx.resolution
    return idx.line_intersects(a, b)


def main():
    gt_items, gt_resolution, bbox = load_annotations("GROUND TRUTH")
    print(f"{len(gt_items)} lines loaded as ground-truth synapses")
    print(f"(Resolution: {tuple(gt_resolution)})")
    print()
    pred_items, pred_resolution, bbox = load_annotations("PREDICTIONS", bbox)
    print(f"{len(pred_items)} lines loaded as predicted synapses")
    print(f"(Resolution: {tuple(pred_resolution)})")
    print()
    if not gt_items or not pred_items:
        sys.exit()
    seg_vol, seg_resolution = load_segmentation()
    print(f"(Resolution: {tuple(seg_resolution)})")

    if bbox is None:
        bbox = {}
        bbox["pointA"] = input_vec3Di("Bounding box start")
        bbox["pointB"] = input_vec3Di("Bounding box end  ")
        bbox["resolution"] = input_vec3Di("...at Resolution  ")

    bbox_start: Vec3D = Vec3D(*bbox["pointA"])
    bbox_end: Vec3D = Vec3D(*bbox["pointB"])
    box_res: Vec3D = Vec3D(*bbox["resolution"])
    bbox = BBox3D.from_points([bbox_start, bbox_end], box_res, epsilon=0)
    idx = VolumetricIndex.from_coords(
        floor(bbox.start / seg_resolution),
        ceil(bbox.end / seg_resolution) + Vec3D(1, 1, 1),
        seg_resolution,
    )
    # Pad the seg data so that we can handle synapses slightly out of bounds.
    idx = idx.padded(Vec3D(64, 64, 32))
    print(f"Reading seg data for: {idx}")
    # seg_data = seg_vol[idx][0]
    s = idx.start
    e = idx.stop
    seg_data = seg_vol[s[0] : e[0], s[1] : e[1], s[2] : e[2]][:, :, :, 0]
    print(f"Seg data shape: {seg_data.shape}")

    # Further filter synapse lines to final bbox
    gt_items = list(filter(lambda x: line_in_bounds(x, gt_resolution, idx), gt_items))
    pred_items = list(filter(lambda x: line_in_bounds(x, pred_resolution, idx), pred_items))
    print(f"Within bounds are {len(gt_items)} GT items and {len(pred_items)} predictions")

    # Analyze presynaptic points
    gt_points = get_points(gt_items, "pointA", gt_resolution)
    pr_points = get_points(pred_items, "pointA", pred_resolution)

    # (print an example, for debugging)
    print(f"{gt_points[0]} maps to seg {lookup_seg_id(gt_points[0], seg_data, idx)}")

    point_segs: Dict[Vec3D, str] = {}
    lookup_seg_ids(gt_points, seg_data, idx, point_segs)
    lookup_seg_ids(pr_points, seg_data, idx, point_segs)
    if not gt_points:
        print("No GT points within bounding box (resolution error?)")
        sys.exit()
    if not pr_points:
        print("No prediction points within bounding box (resolution error?)")
        sys.exit()

    stats = analyze_points(gt_points, pr_points, lambda a, b: point_segs[a] == point_segs[b])
    print_stats(stats, "PRESYNAPTIC SITES")

    # Analyze postsynaptic points
    gt_points = get_points(gt_items, "pointB", gt_resolution)
    pr_points = get_points(pred_items, "pointB", pred_resolution)

    point_segs = {}
    lookup_seg_ids(gt_points, seg_data, idx, point_segs)
    lookup_seg_ids(pr_points, seg_data, idx, point_segs)
    if not gt_points:
        print("No GT points within bounding box (resolution error?)")
        sys.exit()
    if not pr_points:
        print("No prediction points within bounding box (resolution error?)")
        sys.exit()

    stats = analyze_points(gt_points, pr_points, lambda a, b: point_segs[a] == point_segs[b])
    print_stats(stats, "POSTSYNAPTIC SITES")

    # Analyze complete synapses
    # These require a somewhat different procedure — for point_segs, we'll use the
    # string concatenation of IDs for BOTH ends of the line, keyed on the center point.
    point_segs = {}
    for item in gt_items:
        item["center"] = (Vec3D(*item["pointA"]) + Vec3D(*item["pointB"])) / 2
        ids = (
            str(lookup_seg_id(Vec3D(*item["pointA"]) * gt_resolution, seg_data, idx))
            + "-"
            + str(lookup_seg_id(Vec3D(*item["pointB"]) * gt_resolution, seg_data, idx))
        )
        point_segs[item["center"] * gt_resolution] = ids
    for item in pred_items:
        item["center"] = (Vec3D(*item["pointA"]) + Vec3D(*item["pointB"])) / 2
        ids = (
            str(lookup_seg_id(Vec3D(*item["pointA"]) * pred_resolution, seg_data, idx))
            + "-"
            + str(lookup_seg_id(Vec3D(*item["pointB"]) * pred_resolution, seg_data, idx))
        )
        point_segs[item["center"] * pred_resolution] = ids
    gt_points = get_points(gt_items, "center", gt_resolution)
    if not gt_points:
        print("No GT points within bounding box (resolution error?)")
        sys.exit()
    pr_points = get_points(pred_items, "center", pred_resolution)
    if not pr_points:
        print("No prediction points within bounding box (resolution error?)")
        sys.exit()
    breakpoint()
    stats = analyze_points(gt_points, pr_points, lambda a, b: point_segs[a] == point_segs[b])
    print_stats(stats, "SYNAPSES")

    print("\n\nTP:")
    print_as_annotations(stats["tp_points_B"], pred_items, "center", pred_resolution)

    print("\n\nFP:")
    print_as_annotations(stats["fp_points_B"], pred_items, "center", pred_resolution)

    print("\n\nFN:")
    print_as_annotations(stats["fn_points_A"], gt_items, "center", gt_resolution)


if __name__ == "__main__":
    main()
