#!/usr/bin/env python3

import hashlib
import json
import random
import re
import subprocess

################################################################################
# 1) GCS listing and filename parsing
################################################################################


def list_gcs_files(gcs_path):
    """
    Use `gsutil ls <gcs_path>/*` to list chunk files.
    Returns a list of URIs. If it fails or no files found, returns [].
    """
    cmd = ["gsutil", "ls", f"{gcs_path}/*"]
    print(f"[DEBUG] Listing files with: {' '.join(cmd)}")
    try:
        output = subprocess.check_output(cmd, universal_newlines=True)
        lines = output.strip().split("\n")
        uris = [line.strip() for line in lines if line.strip()]
        print(f"[DEBUG] Found {len(uris)} object(s).")
        return uris
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] gsutil ls failed: {e}")
        return []


def parse_filename(filename):
    """
    Expect filenames like: x1-x2_y1-y2_z1-z2 (integers).
    Return (x1, x2, y1, y2, z1, z2) if matched; else None.
    """
    pattern = r"^(\d+)-(\d+)_(\d+)-(\d+)_(\d+)-(\d+)$"
    m = re.match(pattern, filename)
    if not m:
        return None
    return tuple(map(int, m.groups()))  # x1, x2, y1, y2, z1, z2


################################################################################
# 2) Merging logic (coalescing holes up to 2 chunks in X, Y, Z)
################################################################################


def merge_z_intervals(z1_a, z2_a, z1_b, z2_b, dz_ref, hole_factor=2):
    """
    Merge two Z intervals if their gap <= hole_factor * dz_ref.
    Otherwise, return None (meaning they're too far to unify in Z).

    If they unify, we return (z1_merged, z2_merged).

    We assume z1_a < z1_b after sorting if needed.
    """
    # Sort so the first interval is "lower" in z.
    if z1_b < z1_a:
        z1_a, z2_a, z1_b, z2_b = z1_b, z2_b, z1_a, z2_a

    gap = z1_b - z2_a  # distance between the two intervals
    if gap <= hole_factor * dz_ref:
        # unify
        return (z1_a, max(z2_a, z2_b))
    else:
        return None


def bounding_box_close_enough_2chunks(boxA, boxB, dx_ref, dy_ref, dz_ref, hole_factor=2):
    """
    Decide if boxA and boxB can unify in 3D, ignoring holes up to hole_factor * d{xyz}.
    We check:
      1) X: how big the merged bounding box is vs the sum of widths
      2) Y: similarly for heights
      3) Z: we call merge_z_intervals to see if z-gap <= hole_factor * dz_ref
    """
    # X dimension
    x_min = min(boxA["x1"], boxB["x1"])
    x_max = max(boxA["x2"], boxB["x2"])
    widthA = boxA["x2"] - boxA["x1"]
    widthB = boxB["x2"] - boxB["x1"]
    merged_width = x_max - x_min
    gap_x = merged_width - (widthA + widthB)
    if gap_x > hole_factor * dx_ref:
        return False

    # Y dimension
    y_min = min(boxA["y1"], boxB["y1"])
    y_max = max(boxA["y2"], boxB["y2"])
    heightA = boxA["y2"] - boxA["y1"]
    heightB = boxB["y2"] - boxB["y1"]
    merged_height = y_max - y_min
    gap_y = merged_height - (heightA + heightB)
    if gap_y > hole_factor * dy_ref:
        return False

    # Z dimension
    z_merged = merge_z_intervals(
        boxA["z1"], boxA["z2"], boxB["z1"], boxB["z2"], dz_ref, hole_factor
    )
    return z_merged is not None


def merge_box_3d_2chunks(boxA, boxB, dx_ref, dy_ref, dz_ref, hole_factor=2):
    """
    Actually produce the merged box in 3D (union in X, Y).
    For Z, unify intervals if they're within hole_factor * dz_ref.
    We'll call merge_z_intervals again (we know it's not None if close_enough returned True).
    """
    x_min = min(boxA["x1"], boxB["x1"])
    x_max = max(boxA["x2"], boxB["x2"])
    y_min = min(boxA["y1"], boxB["y1"])
    y_max = max(boxA["y2"], boxB["y2"])

    # unify Z intervals
    z_merged = merge_z_intervals(
        boxA["z1"], boxA["z2"], boxB["z1"], boxB["z2"], dz_ref, hole_factor
    )
    z1_merged, z2_merged = z_merged

    return {"x1": x_min, "x2": x_max, "y1": y_min, "y2": y_max, "z1": z1_merged, "z2": z2_merged}


################################################################################
# 3) Converting final bounding boxes to the JSON format with nm coords
################################################################################


def generate_random_id():
    """
    Returns a random hex string, e.g., a 40-char SHA1 hash.
    """
    return hashlib.sha1(str(random.random()).encode("utf-8")).hexdigest()


def bounding_boxes_to_json(bboxes):
    """
    For each final bounding box (in pixel coords):
      {x1, x2, y1, y2, z1, z2}
    produce a JSON record with:
      - pointA = [xA_nm, yA_nm, zA_nm]
      - pointB = [xB_nm, yB_nm, zB_nm]
      - type = "axis_aligned_bounding_box"
      - id = random hex

    We assume voxel size = (128 nm, 128 nm, 40 nm).
    We 'strike' the Z in the middle => (z + 0.5)*40 nm.

    Returns (aabb_list, z_ranges_list):
      - aabb_list is the JSON-friendly array of bounding-box dicts.
      - z_ranges_list is a list of (z1_nm, z2_nm) for the final bboxes.
    """
    NM_X = 128.0
    NM_Y = 128.0
    NM_Z = 1.0

    results = []
    z_ranges = []  # to store (z1_nm, z2_nm) for each final box

    for box in bboxes:
        x1_px, x2_px = box["x1"], box["x2"]
        y1_px, y2_px = box["y1"], box["y2"]
        z1_px, z2_px = box["z1"], box["z2"]

        # Convert X & Y to nm directly
        x1_nm = x1_px * NM_X
        x2_nm = x2_px * NM_X
        y1_nm = y1_px * NM_Y
        y2_nm = y2_px * NM_Y

        # For Z, do (z + 0.5)*40 nm
        z1_nm = (z1_px + 0.5) * NM_Z
        z2_nm = (z2_px + 0.5) * NM_Z

        pointA = [float(min(x1_nm, x2_nm)), float(min(y1_nm, y2_nm)), float(min(z1_nm, z2_nm))]
        pointB = [float(max(x1_nm, x2_nm)), float(max(y1_nm, y2_nm)), float(max(z1_nm, z2_nm))]

        record = {
            "pointA": pointA,
            "pointB": pointB,
            "type": "axis_aligned_bounding_box",
            "id": generate_random_id(),
        }
        results.append(record)

        # Also store the Z range for later printing
        z_ranges.append((pointA[2], pointB[2]))  # minZ, maxZ in nm

    return results, z_ranges


################################################################################
# 4) Main script bringing it all together
################################################################################


def main():
    # The path to your GCS directory
    gcs_path = "gs://sergiy_exp/stroeh_retina/portal/resin_mask_x1/128_128_40"
    hole_factor = 2  # how many chunk-sizes to allow as a hole in X, Y, or Z

    # 1) List chunk files
    uris = list_gcs_files(gcs_path)
    if not uris:
        print("No files found in GCS path.")
        return

    # 2) Parse them into bounding boxes
    bboxes = []
    for uri in uris:
        filename = uri.rsplit("/", 1)[-1]
        parsed = parse_filename(filename)
        if not parsed:
            # skip non-chunk files
            continue
        x1, x2, y1, y2, z1, z2 = parsed
        bboxes.append({"x1": x1, "x2": x2, "y1": y1, "y2": y2, "z1": z1, "z2": z2})

    if not bboxes:
        print("No valid chunk filenames (x1-x2_y1-y2_z1-z2) found.")
        return

    # 3) Determine reference chunk size from the first bounding box
    first = bboxes[0]
    dx_ref = first["x2"] - first["x1"]
    dy_ref = first["y2"] - first["y1"]
    dz_ref = first["z2"] - first["z1"]
    if dx_ref <= 0 or dy_ref <= 0 or dz_ref <= 0:
        print("[WARNING] The first chunk has non-positive size. Merging might be odd.")

    # 4) Iterative merging until no more merges
    changed = True
    while changed:
        changed = False
        used = [False] * len(bboxes)
        new_list = []

        i = 0
        while i < len(bboxes):
            if used[i]:
                i += 1
                continue
            current = bboxes[i]
            used[i] = True
            j = i + 1
            while j < len(bboxes):
                if used[j]:
                    j += 1
                    continue
                candidate = bboxes[j]

                # check if we can unify with up to 2-chunk holes
                if bounding_box_close_enough_2chunks(
                    current, candidate, dx_ref, dy_ref, dz_ref, hole_factor=hole_factor
                ):
                    merged_box = merge_box_3d_2chunks(
                        current, candidate, dx_ref, dy_ref, dz_ref, hole_factor=hole_factor
                    )
                    current = merged_box
                    used[j] = True
                    changed = True
                j += 1

            new_list.append(current)
            i += 1

        # Deduplicate
        tmp = []
        seen = set()
        for b in new_list:
            key = (b["x1"], b["x2"], b["y1"], b["y2"], b["z1"], b["z2"])
            if key not in seen:
                seen.add(key)
                tmp.append(b)
        bboxes = tmp

    # Sort final bounding boxes for stable output
    def bbox_sort_key(b):
        return (b["z1"], b["z2"], b["y1"], b["y2"], b["x1"], b["x2"])

    bboxes_sorted = sorted(bboxes, key=bbox_sort_key)

    # 5) Convert final bounding boxes => JSON with nm coords
    aabbs, z_ranges_nm = bounding_boxes_to_json(bboxes_sorted)

    # 6) Print the JSON array
    print("\n=== FINAL MERGED BOUNDING BOXES (in nm) ===\n")
    print(json.dumps(aabbs, indent=2))

    # 7) Also print all final Z ranges in ascending order
    #    z_ranges_nm is a list of (z_min_nm, z_max_nm). Sort by z_min_nm.
    z_ranges_sorted = sorted(z_ranges_nm, key=lambda r: r[0])
    print("\n=== FINAL Z RANGES (nm) ===")
    for (zmin, zmax) in z_ranges_sorted:
        print(f"  Z=({zmin:.2f} - {zmax:.2f}) nm")


if __name__ == "__main__":
    main()
