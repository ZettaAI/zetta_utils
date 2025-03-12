"""
This script looks up the segment (supervoxel) IDs for the points in a CAVE
table, and stuffs those ids into the supervoxel table.  Should be general
enough to work for any such table.

Command-line options:
    --dry-run: do everything except actually write to the DB
"""
import argparse
import json
import sys
from collections import defaultdict
from math import floor
import numpy as np
from typing import Dict, List, Tuple

from google.cloud import storage
from sqlalchemy import create_engine
from sqlalchemy import text as sql
from sqlalchemy.engine import URL

from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex
from zetta_utils.layer.volumetric.cloudvol import build_cv_layer
from zetta_utils.layer.volumetric.precomputed import PrecomputedInfoSpec

# Database connection parameters
DB_USER = "postgres"
DB_PASS = ""  # Put DB password here
DB_NAME = "dacey_human_fovea"
DB_HOST = "127.0.0.1"  # Local proxy address; run Cloud SQL Auth Proxy
DB_PORT = 5432  # Default PostgreSQL port

# pylint: disable=global-statement

# Globals
engine = None
base_table = ""  # e.g. cell_points
supervox_table = ""  # e.g. cell_points__dacey_human_fovea_2404
point_field_name = "pt_position"
supervoxel_field_name = "pt_supervoxel_id"
point_resolution = Vec3D(0, 0, 0)
seg_layer = None
seg_bounds = None
args = None

seg_resolution = Vec3D(0, 0, 0)
seg_chunk_size = Vec3D(0, 0, 0)
seg_offset = Vec3D(0, 0, 0)

# Conversion factors for going from one resolution to another
syn_to_seg = Vec3D(0.0, 0.0, 0.0)

engine = None
chunk = None
chunk_bounds = None

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


def chunk_index_for_point(seg_point):
    """
    Return a VolumetricIndex representing the chunk containing the
    given (Vec3D) point, in segmentation voxel coordinates.
    """
    offset_point = seg_point - seg_offset
    chunk_count = floor(offset_point / seg_chunk_size)
    chunk_start = seg_offset + chunk_count * seg_chunk_size
    return VolumetricIndex.from_coords(chunk_start, chunk_start + seg_chunk_size, seg_resolution)


def bin_index_for_point(seg_point):
    return floor((seg_point - seg_offset) / seg_chunk_size)


def bin_points(to_do):
    """
    Bins points based on their coordinates.

    Args:
        to_do: List of seg point dicts (including "id" and "point")

    Returns:
        Dictionary mapping bin indices to list of points
    """
    # Use defaultdict to automatically create new bins as needed
    bins = defaultdict(list)

    for point in to_do:
        bin_idx = bin_index_for_point(point["point"])
        bins[bin_idx].append(point)

    # Convert defaultdict to regular dict for return
    return dict(bins)


def row_to_dict(row: Tuple, keys: Tuple[str, ...]) -> Dict:
    """
    Convert a database row containing point data into a dict containing
    'id' and 'point' (Vec3Ds in segmentation volume coordinates),
    and 'sv_id' set to None.
    """
    row_dict = dict(zip(keys, row))
    result = {
        "id": row_dict["id"],
        "point": Vec3D(
            float(row_dict["pt_x"]), float(row_dict["pt_y"]), float(row_dict["pt_z"])
        )
        * syn_to_seg,
        "sv_id": None
    }
    return result


def read_points(conn, table_name, where_clause) -> List[Dict]:
    """
    Read point records, extracting coordinates from PostGIS points
    and converting them immediately into segmentation volume coordinates.

    Returns:
        List of dictionaries containing id and coordinates for pre/post points
    """
    query = sql(
        f"""
        SELECT
            id,
            ST_X({point_field_name}) as pt_x,
            ST_Y({point_field_name}) as pt_y,
            ST_Z({point_field_name}) as pt_z
        FROM {table_name}
    """
        + (f"WHERE {where_clause}" if where_clause else "")
    )
    record_set = conn.execute(query)
    keys = record_set.keys()
    result = []
    for rec in record_set:
        result.append(row_to_dict(rec, keys))
    return result


def bulk_insert_sv_ids(items: List[Dict], table_name: str, batch_size: int = 1000) -> None:
    """
    Bulk insert supervoxel IDs into the database.

    Parameters:
        items: List of dictionaries, each containing 'id', 'sv_id', and 'segB'
        table_name: Name of the target table
        batch_size: Number of rows to insert in each batch (default 1000)
    """

    def create_batch_query(batch_items: List[Dict]) -> str:
        value_entries = []
        for item in batch_items:
            value_entries.append(f"({item['id']},{item['sv_id']})")

        values_clause = ",\n".join(value_entries)

        return f"""
            INSERT INTO {table_name} (id, {supervoxel_field_name})
            VALUES
                {values_clause}
            ON CONFLICT (id) DO UPDATE
            SET {supervoxel_field_name} = EXCLUDED.{supervoxel_field_name};
        """

    assert engine is not None
    with engine.connect() as conn:  # type: ignore[unreachable]
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            query = sql(create_batch_query(batch))
            if args.dry_run:
                print("Dry Run mode; SQL that we're not executing:")
                print(query)
            else:
                conn.execute(query)
            print(f"Processed {min(i + batch_size, len(items))}/{len(items)} items")

        if not args.dry_run:
            conn.commit()
            print(f"Committed {len(items)} new rows to {table_name}")


def input_vec3D(prompt="", default=None):
    while True:
        s = input(prompt + (f" [{default.x}, {default.y}, {default.z}]" if default else "") + ": ")
        if s == "" and default:
            return default
        try:
            x, y, z = map(float, s.replace(",", " ").split())
            return Vec3D(x, y, z)
        except:  # pylint: disable=bare-except
            print("Enter x, y, and z values separated by commas or spaces.")


def input_vec3Di(prompt="", default=None):
    v = input_vec3D(prompt, default)
    return round(v)


def lookup_segment_id(seg_point: Vec3D, load_data_if_needed: bool = False):
    """
    Try to look up the given point (in global segmentation coordinates) in
    our currently loaded chunk.  If the point is not within the current
    chunk_bounds, then:
        if load_data_if_needed is True, load the right chunk;
        else return None.
    """
    # pylint: disable-next=global-statement
    global chunk, chunk_bounds
    if chunk is None or not chunk_bounds.contains(seg_point):  # type: ignore[unreachable]
        if not load_data_if_needed:
            return None
        chunk_bounds = chunk_index_for_point(seg_point)
        print(f'Loading new chunk: {chunk_bounds.bbox}...', end='', flush=True)
        assert seg_layer is not None
        assert seg_layer[chunk_bounds] is not None  # type: ignore[unreachable]
        chunk = seg_layer[chunk_bounds][0]
        print('Chunk loaded.')
    relative_point = floor(seg_point - chunk_bounds.start)  # type: ignore[unreachable]
    result = chunk[relative_point[0], relative_point[1], relative_point[2]]
    return result


def process_points(to_do):
    total_to_do = len(to_do)
    print(f"Looking up supervoxel IDs for {total_to_do} points...")
    done = []
    while to_do:
        point = to_do.pop()
        point["sv_id"] = lookup_segment_id(point["point"], True)
        done.append(point)

        if len(done) > 500:
            print(f"Saving done records to {supervox_table}...")
            bulk_insert_sv_ids(done, supervox_table)
            done = []
            
    if done:
        print(f"Saving last done records to {supervox_table}...")
        bulk_insert_sv_ids(done, supervox_table)
    print(f"Finished processing {total_to_do} points!")


def main():
    global args, engine, base_table, supervox_table, point_resolution
    global seg_layer, seg_bounds, seg_resolution, seg_chunk_size, seg_offset
    global syn_to_seg

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print("Dry Run mode active.")

    DB_PASS = input(f"{DB_NAME} password: ")

    # Create the connection URL
    connection_url = URL.create(
        drivername="postgresql+psycopg2",
        username=DB_USER,
        password=DB_PASS,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
    )

    # Create the engine
    engine = create_engine(
        connection_url, connect_args={"connect_timeout": 10}
    )  # 10 seconds timeout

    # Try to connect, just to be sure we can
    try:
        print(f"Connecting to {DB_NAME} at {DB_HOST}:{DB_PORT}...")
        with engine.connect() as _:
            print("Successfully connected to the database!")

    # pylint: disable-next=broad-exception-caught
    except Exception as e:
        print("Error connecting to the database:")
        print(e)
        sys.exit()

    # Get parameters from the user
    while True:
        supervox_table = input("Supervoxel table name: ")
        if len(supervox_table.split("__")) == 2:
            break
        print(
            "This should look something like, for example: "
            "cell_points__dacey_human_fovea_2404"
        )
    base_table = supervox_table.split("__")[0]
    point_resolution = input_vec3Di("Synapse voxel scale (resolution)")

    print()
    seg_path = input("Supervoxel volume path: ")
    data = read_gcs_json(seg_path + "/info")
    scale0 = data["scales"][0]
    seg_resolution = Vec3D(*scale0["resolution"])
    seg_chunk_size = Vec3D(*scale0["chunk_sizes"][0])
    seg_offset = Vec3D(*scale0["voxel_offset"])
    print(f"   Supervoxel resolution: {seg_resolution}")
    print(f"   Supervoxel chunk size: {seg_chunk_size}")
    print(f"   Supervoxel offset:     {seg_offset}")

    syn_to_seg = point_resolution / seg_resolution
    print(f"   Conversion factor from point to segmentation: {syn_to_seg}")

    # Load the segmentation layer
    seg_layer, seg_bounds = load_volume(seg_path)

    to_do: List[dict] = []
    with engine.connect() as conn1:
        print("Reading points...")
        where_clause = f"""
NOT EXISTS (SELECT 1 FROM {supervox_table} b WHERE b.id = {base_table}.id
AND b.{supervoxel_field_name} <> 0);
"""
        to_do = read_points(conn1, base_table, where_clause)

    print(f"Read {len(to_do)} points from {base_table}")
    print("Sorting points into chunks...")
    bins = bin_points(to_do)
    print(f"Divided them into {len(bins)} bins")
    sorted_indexes = sorted(bins.keys())
    for i, bin_idx in enumerate(sorted_indexes):
        print()
        print(f"STARTING BIN {i}/{len(sorted_indexes)} (bin {bin_idx})...")
        process_points(bins[bin_idx])


if __name__ == "__main__":
    main()
