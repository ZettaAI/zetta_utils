"""
This script takes a set of points placed inside organelles (nuclei, mitochondria, or
really anything our segmentation process has segmented inside a cell).  It figures
out what cell that organelle is contained in, and then issues a chunkedgraph merge
to join the organelle into the cell.
"""

import argparse
import json
import readline
import struct
import scipy
import sys
import numpy as np
from google.cloud import storage
from caveclient import CAVEclient, chunkedgraph
from cloudvolume import CloudVolume
from math import floor
from typing import Any, Sequence
from binascii import unhexlify
from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex
from zetta_utils.layer.volumetric.cloudvol import build_cv_layer
from zetta_utils.layer.volumetric.precomputed import PrecomputedInfoSpec
from zetta_utils.geometry.mask_center import interior_point

# Globals
point_resolution = Vec3D(0, 0, 0)
seg_vol = None
seg_bounds = None
args = None
seg_resolution = Vec3D(0, 0, 0)
seg_chunk_size = Vec3D(0, 0, 0)
seg_offset = Vec3D(0, 0, 0)
inp_to_seg = Vec3D(0.0, 0.0, 0.0)
chunk = None
chunk_bounds = None
cg_client = None
line_annotations = []   # lines representing merges
point_annotations = []  # points representing failed/aborted merges
pcg_table_name = ''

def verify_cave_auth() -> bool:
    # pylint: disable-next=global-statement
    global client
    client = CAVEclient(
        datastack_name="dacey_human_fovea", 
        server_address='https://proofreading.zetta.ai'
    )
    try:
        # pylint: disable-next=pointless-statement    # bite me, pylint
        client.state
        return  # no exception?  All's good!
    # pylint: disable-next=bare-except
    except:
        pass
    print("Authentication needed.")
    print("Go to: https://proofreading.zetta.ai/sticky_auth/settings/tokens")
    print("   or: https://global.daf-apis.com/auth/api/v1/create_token")
    token = input("Enter token: ")
    client.auth.save_token(token=token)


def parse_wkb_hex_point(hex_str: str) -> tuple:
    """Parse a WKB hex string representing a point."""
    binary = unhexlify(hex_str)
    # Skip first 5 bytes (endianness and geometry type)
    # Then read two or three 8-byte doubles
    try:
        x, y, z = struct.unpack_from("ddd", binary, offset=5)
        return x, y, z
    except struct.error:
        # Fall back to 2D if 3D fails
        try:
            x, y = struct.unpack_from("dd", binary, offset=5)
            return x, y, 0
        except struct.error as exc:
            raise ValueError("Invalid WKB hex string") from exc

def parse_point(point_string: str) -> Vec3D:
    """This function attempts to parse a string as a 3D point.
    It supports any combination of spaces, commas, or vertical
    bars as field separators, and strips extra separators from
    the start and end of the string.  It also supports WKB hex
    strings, as you might get from a DB if you don't unpack
    a geospatial point.
    """
    s = point_string.replace("|", " ").replace(",", " ")
    fields = s.split()
    if len(fields) == 0:
        return None
    if len(fields) == 1:
        return Vec3D(*parse_wkb_hex_point(fields[0]))
    elif len(fields) == 3:
        return Vec3D(*map(float, fields))
    else:
        raise ValueError("Invalid point input: " + point_string)


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


def load_volume(path, scale_index=0):
    """
    Load a CloudVolume given the path, and optionally, which scale (resolution) is desired.
    Return the CloudVolume, and a BBox3D describing the data bounds.
    """
    vol = CloudVolume(path, mip=scale_index)
    info = vol.info
    assert info is not None
    scale = info["scales"][scale_index]
    resolution = scale["resolution"]
    start_coord = scale["voxel_offset"]
    size = scale["size"]
    end_coord = [a + b for (a, b) in zip(start_coord, size)]
    bounds = BBox3D.from_coords(start_coord, end_coord)
    return vol, bounds

def safe_get_block(vol, x_slice, y_slice, z_slice):
    """
    Get a 3D block of data out of the given CloudVolume (ignoring the 4th dimension),
    padding with zeros as needed on the sides to avoid any OutOfBoundsException.
    """
    min_coords = tuple(vol.voxel_offset)
    max_coords = tuple(vol.voxel_offset[i] + vol.shape[i] for i in (0, 1, 2))

    # Compute the requested block shape
    x_range = range(x_slice.start or min_coords[0], x_slice.stop or max_coords[0])
    y_range = range(y_slice.start or min_coords[1], y_slice.stop or max_coords[1])
    z_range = range(z_slice.start or min_coords[2], z_slice.stop or max_coords[2])

    output_shape = (len(x_range), len(y_range), len(z_range))

    # Prepare an array of zeros for padding
    padded_block = np.zeros(output_shape, dtype=vol.dtype)

    # Compute valid slice ranges
    x_start = max(x_slice.start or min_coords[0], min_coords[0])
    x_stop = min(x_slice.stop or max_coords[0], max_coords[0])
    y_start = max(y_slice.start or min_coords[1], min_coords[1])
    y_stop = min(y_slice.stop or max_coords[1], max_coords[1])
    z_start = max(z_slice.start or min_coords[2], min_coords[2])
    z_stop = min(z_slice.stop or max_coords[2], max_coords[2])

    # Compute where to place the valid data in the output array
    x_offset = x_start - (x_slice.start or min_coords[0])
    y_offset = y_start - (y_slice.start or min_coords[1])
    z_offset = z_start - (z_slice.start or min_coords[2])

    valid_block = vol[x_start:x_stop, y_start:y_stop, z_start:z_stop].squeeze(axis=-1)

    # Insert the valid block into the zero-padded array
    padded_block[
        x_offset:x_offset + valid_block.shape[0],
        y_offset:y_offset + valid_block.shape[1],
        z_offset:z_offset + valid_block.shape[2]
    ] = valid_block

    return padded_block

def chunk_index_for_point(seg_point):
    """
    Return a VolumetricIndex representing the chunk containing the
    given (Vec3D) point, in segmentation voxel coordinates.
    """
    offset_point = seg_point - seg_offset
    chunk_count = floor(offset_point / seg_chunk_size)
    chunk_start = seg_offset + chunk_count * seg_chunk_size
    return VolumetricIndex.from_coords(chunk_start, chunk_start + seg_chunk_size, seg_resolution)


def lookup_ids(seg_point: Vec3D):
    """
    Try to look up the given point (in global segmentation coordinates) in
    our graphene volume.  Return both the supervoxel ID and the root ID.
    """
    # pylint: disable-next=global-statement
    sv_id = seg_vol[seg_point[0], seg_point[1], seg_point[2]].item()
    root_id = seg_vol.get_roots(sv_id).item()
    return sv_id, root_id


def get_id_blocks(center_point: Vec3D, size: Vec3D=Vec3D(256,256,8)):
    """
    Return a 3D block of supervoxel IDs and root IDs around the given point.
    """
    half_size = size / 2
    start_pos = floor(center_point - half_size)
    end_pos = floor(center_point + half_size)
    sv_block = safe_get_block(seg_vol, 
                              slice(start_pos[0], end_pos[0]), 
                              slice(start_pos[1], end_pos[1]),
                              slice(start_pos[2], end_pos[2]))
    root_block = seg_vol.get_roots(sv_block.reshape(-1)).reshape(sv_block.shape)
    return sv_block, root_block

def most_common_nonzero(data: np.ndarray, return_count=False):
    """
    Returns the most common nonzero value in the input array.
    If no nonzero values are present, returns None.
    """
    unique_values, counts = np.unique(data, return_counts=True)
    if unique_values[0] == 0:
        unique_values = unique_values[1:]
        counts = counts[1:]

    if len(unique_values) > 0:
        # For debugging: sort and display top few values
        # sort_indexes = np.argsort(counts)
        # counts = counts[sort_indexes]
        # unique_values = unique_values[sort_indexes]
        # print(f"Top IDs and counts: {list(zip(unique_values[-10:], counts[-10:]))}")

        idx = np.argmax(counts)
        if return_count:
            return unique_values[idx], counts[idx]
        return unique_values[idx]
    else:
        if return_count:
            return None, 0
        return None

def touches_XY_edge_of_block(value, data_block):
    """
    Return whether the given value is found on any X or Y face of the given block of data.
    This can be used, for example, to determine whether a cell found at the center
    extends to the edge (and probably beyond) of the data window.
    """
    return any(
        np.any(face == value)
        for face in [
            data_block[0, :, :],  # left face
            data_block[-1, :, :],  # right face
            data_block[:, 0, :],  # top face
            data_block[:, -1, :],  # bottom face
        ]
    )


def abbreviate(long_id, max_length=11):
    s = str(long_id)
    if len(s) > max_length:
        s = s[:max_length//2] + "…" + s[-max_length//2:]
    return s

def process_line(line: str):
    """
    Parse the given line to a 3D point, then check in the segmentation layer.  If the
    segment containing this point is surrounded by only one other point, then construct
    a merge between the two, using two points across an arbitrary point on the border.
    """
    point = None
    try:
        point = parse_point(line)  # get point, in input coordinates
    except:
        pass
    if point is None:
        return
    print(f"\n{tuple(point)} ...")
    point *= inp_to_seg   # convert to segmentation coordinates
    size = Vec3D(512, 512, 8)
    sv_block, root_block = get_id_blocks(point, size)
    center = tuple(s // 2 for s in size)
    central_root_id = root_block[center]
    if central_root_id == 0:
        print(f"Got 0 central root ID; skipping")
        return
    if touches_XY_edge_of_block(central_root_id, root_block):
        print(f"{central_root_id} at {round(point/inp_to_seg)} extends beyond data window; skipping")
        return
    print(f"central_root_id = {central_root_id}")
    dpos = floor(point - size/2)
    point_offset = Vec3D(0, 0, -0.4) # (to work around NG's off-by-0.5-in-Z issue)

    mask = (root_block == central_root_id)
    dilated_mask = scipy.ndimage.binary_dilation(mask)
    border_mask = dilated_mask & ~mask
    border_roots = root_block[border_mask]
    neighbor_root_id, neighbor_count = most_common_nonzero(border_roots, return_count=True)
    total_neighbors = border_mask.sum()
    percent_covered = round(100 * neighbor_count / total_neighbors)
    print(f"{total_neighbors} total neighbors; biggest is {neighbor_root_id} with {neighbor_count} ({percent_covered}%)")
    threshold = int(args.pct)
    if percent_covered < threshold:
        print(f"That's not dominant enough ({percent_covered} < {threshold}).  Bailing out.")
        if args.anno:
            point_annotations.append(
                "{" +
                f'"point":{list(round(point / inp_to_seg + point_offset))},'
                f'"description":"{neighbor_root_id}: {percent_covered} < {threshold}",'
                '"type":"point",'
                f'"id":"{central_root_id}"'
                "}"
            )
        return

    # Find a border between our central value and the dominant neighbor value.
    # And let's do this only in the same Z plane as the point.
    z = center[2]
    root_slice = root_block[:, :, z]
    sv_slice = sv_block[:, :, z]
    border_mask = border_mask[:, :, z]
    neighbor_sv_id = most_common_nonzero(sv_slice[(root_slice == neighbor_root_id) & border_mask])
    neighbor_pos = interior_point(sv_slice == neighbor_sv_id)
    print(f'Border neighbor sv: {neighbor_sv_id}, with center at {neighbor_pos}')

    neighbor_sv_mask = (sv_slice == neighbor_sv_id)
    neighbor_border_mask = scipy.ndimage.binary_dilation(neighbor_sv_mask, iterations=2) & ~neighbor_sv_mask
    proximal_svs = sv_slice[(root_slice == central_root_id) & neighbor_border_mask]
    if proximal_svs.any():
        central_sv_id = most_common_nonzero(proximal_svs)
        print(f'A central supervoxel next to that is {central_sv_id}')
    else:
        central_sv_id = sv_slice[center[0], center[1]]
        print(f"WTF? Couldn't find any nearby central SVs; using {central_sv_id} at input point")
    central_pos = interior_point(sv_slice == central_sv_id)
    print(f'And THAT has a center at {central_pos}, with SV {sv_slice[central_pos]} and root {root_slice[central_pos]}')

    # Calculate our endpoints, in input coordinates
    neighbor_point = round(Vec3D(dpos[0] + neighbor_pos[0], dpos[1] + neighbor_pos[1], point[2]) / inp_to_seg)
    central_point = round(Vec3D(dpos[0] + central_pos[0], dpos[1] + central_pos[1], point[2]) / inp_to_seg)
    print(f'Merge line: {tuple(neighbor_point)} - {tuple(central_point)}')
    
    if not args.dry_run:
        do_merge(central_sv_id, central_point, neighbor_sv_id, neighbor_point)

    if args.anno:
        line_annotations.append(
            "{" +
            f'"pointA":{list(central_point + point_offset)},'
            f'"pointB":{list(neighbor_point + point_offset)},'
            f'"description":"{abbreviate(central_root_id)} - {abbreviate(neighbor_root_id)}",'
            '"type":"line",'
            f'"id":"{central_root_id}"'
            "}"
        )

def do_merge(sv1: int, point1: Vec3D, sv2: int, point2: Vec3D):
    global cg_client, pcg_table_name
    if cg_client is None:
        cg_client = chunkedgraph.ChunkedGraphClient(
            server_address='https://data.proofreading.zetta.ai', 
            table_name=pcg_table_name, 
            auth_client=client.auth
        )
    print(f"Merging SVs {sv1}-{sv2}, at points {point1}-{point2}, in {pcg_table_name}")
    cg_client.do_merge((sv1, sv2), (point1, point2), point_resolution)

def process_stdin():
    for line in sys.stdin:
        process_line(line.strip())
        
def main():
    global args, engine, base_table, supervox_table, point_resolution
    global seg_vol, seg_bounds, seg_resolution, seg_chunk_size, seg_offset
    global inp_to_seg, line_annotations, point_annotations, pcg_table_name

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--resolution", help="resolution of point coordinates", required=True)
    parser.add_argument("--seg", help="segmentation volume path", required=True)
    parser.add_argument("--sv", help="supervoxel volume path")
    parser.add_argument("--pct", default="90", help="percent of border which must be touched to merge")
    parser.add_argument("--dry-run", action="store_true", help="if true, print merges without actually doing them")
    parser.add_argument("--anno", action="store_true", help="if true, output lines/points representing the results")
    args = parser.parse_args()
    if args.dry_run:
        print("Dry Run mode active.")

    pcg_table_name = args.seg.split('/')[-1]
    print(f'PCG table: {pcg_table_name}')
    point_resolution = parse_point(args.resolution)
    print(f"Input point resolution: {tuple(point_resolution)}")

    # HACK for testing:
    # do_merge(72902569395294162, (6435, 5514, 2682), 72902569395292075, (6444, 5520, 2682))
    # sys.exit()

    print()
    seg_path = args.seg #input("Segmentation volume path: ")
    seg_vol, seg_bounds = load_volume(seg_path, 1)
    seg_resolution = Vec3D(*seg_vol.resolution)
    seg_chunk_size = Vec3D(*seg_vol.chunk_size)
    seg_offset = Vec3D(*seg_vol.voxel_offset)
    print(f"   Supervoxel resolution: {seg_resolution}")
    print(f"   Supervoxel chunk size: {seg_chunk_size}")
    print(f"   Supervoxel offset:     {seg_offset}")

    inp_to_seg = point_resolution / seg_resolution
    print(f"   Conversion factor from point to segmentation: {inp_to_seg}")

    line_annotations = []
    point_annotations = []
    process_stdin()

    if args.anno:
        print(f"\n{len(line_annotations)} merge lines, in resolution {tuple(point_resolution)}:\n")
        print(",\n".join(line_annotations))

        print(f"\n{len(point_annotations)} failed merges, in resolution {tuple(point_resolution)}:\n")
        print(",\n".join(point_annotations))


if __name__ == "__main__":
    verify_cave_auth()
    main()
