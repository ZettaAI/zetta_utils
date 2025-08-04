"""
This script compares predicted synaptic clefts to ground truth clefts,
reporting precision, recall, and F1 score.
"""

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

import cc3d
import nglui
import numpy as np
from caveclient import CAVEclient
from cloudfiles import CloudFile
from cloudvolume import CloudVolume
from google.cloud import storage
from scipy.ndimage import center_of_mass
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex
from zetta_utils.layer.volumetric.annotation.build import build_annotation_layer
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


def load_segmentation(purpose: str):
    global source_path
    layer = None
    lineItems = None
    boxItem = None
    resolution = None
    print(f"Enter Neuroglancer state link or ID, or a GS file path for {purpose} segmentation:")
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
        print(f"Select segmentation layer to use for {purpose}:")
        seg_layer_name = get_segmentation_layer_name(state)
        layer_data = nglui.parser.get_layer(state, seg_layer_name)
        source_path = layer_data["source"]
        if isinstance(source_path, dict):  # type: ignore[unreachable]
            source_path = source_path["url"]  # type: ignore[unreachable]
        resolution = get_resolution(layer_data)
    if "/|neuroglancer-precomputed:" in source_path:
        source_path = source_path.split("/|neuroglancer-precomputed:")[0]
    print(f"Loading segmentation from {source_path}...")

    return build_cv_layer(source_path, default_desired_resolution=resolution), resolution

    # seg_cvl = CloudVolume(source_path)
    # seg_cvl.agglomerate = True  # get root IDs, not supervoxel IDs
    # seg_cvl.fill_missing = True # and don't crash for no good reason
    # return seg_cvl, Vec3D(*seg_cvl.resolution)


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

    max_distance = 10  # (voxels)

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
    result["mean_dist"] = sum(distances) / len(distances)
    result["median_dist"] = distances[len(distances) // 2]
    result["max_dist"] = distances[-1]
    result["min_dist"] = distances[0]

    # Detailed match info, for debugging
    print(f"Found {len(matches)} matches, such as: {matches[:3]}")
    try:
        result["matches_list"] = matches
        result["tp_points_A"] = [points_A[i] for i, _ in matches]
        result["tp_points_B"] = [points_B[i] for _, i in matches]
        result["fp_points_B"] = list([p for p in points_B if p not in result["tp_points_B"]])
        result["fn_points_A"] = list([p for p in points_A if p not in result["tp_points_A"]])
    except:
        pass
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


def print_as_annotations(points, items, offset: Vec3D):
    result = []
    id = 1
    for p in points:
        pos = Vec3D(*p) + offset
        result.append("{" + f""""type": "point", "id": "{id}", "point": {list(pos)}""" + "}")
        id += 1
    print(",\n".join(result))


# def centroid(binary_image_array):
#     # Find the interior pixels (all pixels with a value of 1),
#     # and average to find the centroid
#     interior_coords = np.argwhere(binary_image_array == 1)
#     centroid = np.mean(interior_coords, axis=0)
#     return centroid
def centroid(binary_image_array):
    coords = np.transpose(np.nonzero(binary_image_array))
    return coords.mean(axis=0) if coords.size else np.array([np.nan] * binary_image_array.ndim)


def find_clusters(
    data: np.ndarray, offset: Vec3D, within_bounds: VolumetricIndex
) -> Dict[np.integer, Vec3D]:
    """
    Find the centroid of each cluster in the data using numpy operations.
    Discard any that (after adding offset) fall outside of the given bounds.
    Return a dict mapping each remaining cluster ID to its centroid.
    """
    if data.dtype == np.uint8:
        # Run connected components on this data to get clusters
        data = cc3d.connected_components(data, connectivity=26)
    # Get unique cluster IDs (excluding 0)
    labels = np.unique(data)
    labels = labels[labels != 0]  # Skip background
    print(f"Finding centroids for {len(labels)} clusters")
    centroids = center_of_mass(np.ones_like(data), labels=data, index=labels)
    d = dict(zip(labels, centroids))
    for k in labels:
        p = Vec3D(*d[k]) + offset
        if not within_bounds.contains(p):
            del d[k]
    print(f"Found {len(d)} centroids within bounds")
    return d


if __name__ == "__main__":  # def main():
    pred_vol, pred_resolution = load_segmentation("PREDICTIONS")
    gt_vol, gt_resolution = load_segmentation("GROUND TRUTH")

    bbox_start: Vec3D | None = input_vec3Di("Bounding box start")
    bbox_end: Vec3D | None = input_vec3Di("Bounding box end  ")
    bbox_res: Vec3D | None = input_vec3Di("...at Resolution  ")

    if bbox_start is None or bbox_end is None or bbox_res is None:
        print("No bounding box specified")
        sys.exit()

    idx = VolumetricIndex.from_coords(bbox_start, bbox_end, bbox_res)

    # Pad the seg data so that we can handle synapses slightly out of bounds.
    padded_idx = idx.padded(Vec3D(16, 16, 8))
    print(f"Reading seg data for: {idx}")
    # seg_data = seg_vol[idx][0]
    s = padded_idx.start
    e = padded_idx.stop
    pred_data = pred_vol[bbox_res, s[0] : e[0], s[1] : e[1], s[2] : e[2]][0]
    gt_data = gt_vol[bbox_res, s[0] : e[0], s[1] : e[1], s[2] : e[2]][0]

    # Get the centroids of the clusters in the data
    pred_centroids = find_clusters(pred_data, padded_idx.start, idx)
    gt_centroids = find_clusters(gt_data, padded_idx.start, idx)
    print(f"Found {len(pred_centroids)} predicted and {len(gt_centroids)} ground truth synapses")

    # Analyze predictions vs ground truth
    stats = analyze_points(list(gt_centroids.values()), list(pred_centroids.values()))
    print_stats(stats, "PRESYNAPTIC SITES")

    print("\n\nFP:")
    print_as_annotations(stats["fp_points_B"], pred_centroids, padded_idx.start)

    print("\n\nFN:")
    print_as_annotations(stats["fn_points_A"], gt_centroids, padded_idx.start)


# if __name__ == "__main__":
#     main()
