"""
This script looks up the segment (supervoxel) IDs for the endpoints of synapses
which are not already in the supervoxel table, and stuffs those ids into the
supervoxel table, in a CAVE annotation DB.

Command-line options:
    --dry-run: do everything except actually write to the DB
"""
import argparse
import json
import sys
from collections import defaultdict
from getpass import getpass
from math import floor
from typing import Dict, List, Tuple

from google.cloud import storage
from sqlalchemy import create_engine
from sqlalchemy import text as sql
from sqlalchemy.engine import URL

from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.precomputed import PrecomputedInfoSpec
from zetta_utils.layer.volumetric import VolumetricIndex
from zetta_utils.layer.volumetric.cloudvol import build_cv_layer

# Database connection parameters
DB_USER = "postgres"
DB_PASS = ""  # We'll ask for this below
DB_NAME = "dacey_human_fovea"
DB_HOST = "127.0.0.1"  # Local proxy address; run Cloud SQL Auth Proxy
DB_PORT = 5432  # Default PostgreSQL port

# pylint: disable=global-statement

# Globals
engine = None
synapse_table = ""  # e.g. ipl_ribbon_synapses
supervox_table = ""  # e.g. ipl_ribbon_synapses__dacey_human_fovea_2404
synapse_resolution = Vec3D(0, 0, 0)
seg_layer = None
seg_bounds = None

seg_resolution = Vec3D(0, 0, 0)
seg_chunk_size = Vec3D(0, 0, 0)
seg_offset = Vec3D(0, 0, 0)

# Conversion factors for going from one resolution to another
syn_to_seg = Vec3D(0.0, 0.0, 0.0)

engine = None
chunk = None
chunk_bounds = None

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()
if args.dry_run:
    print("Dry Run mode active.")


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


def bin_synapses(to_do):
    """
    Bins synapses based on their pointA and pointB coordinates.  Note that each
    synapse may appear in one or two bins, but it should never be included
    twice.

    Args:
        to_do: List of synapse dicts

    Returns:
        Dictionary mapping bin indices to list of synapses
    """
    # Use defaultdict to automatically create new bins as needed
    bins = defaultdict(list)

    for synapse in to_do:
        # Get bin indices for both points
        bin_a = bin_index_for_point(synapse["pointA"])
        bin_b = bin_index_for_point(synapse["pointB"])

        # Add synapse to bin_a
        bins[bin_a].append(synapse)

        # If points are in different bins, add to bin_b as well
        if bin_b != bin_a:
            bins[bin_b].append(synapse)

    # Convert defaultdict to regular dict for return
    return dict(bins)


def row_to_dict(row: Tuple, keys: Tuple[str, ...]) -> Dict:
    """
    Convert a database row containing synapse data into a dict containing
    'id', 'pointA' and 'pointB' (Vec3Ds in segmentation volume coordinates),
    and 'segA' and 'segB' set to None.
    """
    row_dict = dict(zip(keys, row))
    result = {
        "id": row_dict["id"],
        "pointA": Vec3D(
            float(row_dict["pre_x"]), float(row_dict["pre_y"]), float(row_dict["pre_z"])
        )
        * syn_to_seg,
        "pointB": Vec3D(
            float(row_dict["post_x"]), float(row_dict["post_y"]), float(row_dict["post_z"])
        )
        * syn_to_seg,
        "segA": None,
        "segB": None,
    }
    return result


def read_synapses(conn, table_name, where_clause) -> List[Dict]:
    """
    Read synapse records, extracting coordinates from PostGIS points
    and converting them immediately into segmentation volume coordinates.

    Returns:
        List of dictionaries containing id and coordinates for pre/post points
    """
    query = sql(
        f"""
        SELECT
            id,
            ST_X(pre_pt_position) as pre_x,
            ST_Y(pre_pt_position) as pre_y,
            ST_Z(pre_pt_position) as pre_z,
            ST_X(post_pt_position) as post_x,
            ST_Y(post_pt_position) as post_y,
            ST_Z(post_pt_position) as post_z
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


def bulk_insert_seg_ids(items: List[Dict], table_name: str, batch_size: int = 1000) -> None:
    """
    Bulk insert supervoxel (segment) IDs into the database.

    Parameters:
        items: List of dictionaries, each containing 'id', 'segA', and 'segB'
        table_name: Name of the target table
        batch_size: Number of rows to insert in each batch (default 1000)
    """

    def create_batch_query(batch_items: List[Dict]) -> str:
        value_entries = []
        for item in batch_items:
            value_entries.append(f"({item['id']},{item['segA']},NULL,{item['segB']},NULL)")

        values_clause = ",\n".join(value_entries)

        return f"""
            INSERT INTO {table_name} (id, pre_pt_supervoxel_id, pre_pt_root_id, post_pt_supervoxel_id, post_pt_root_id)
            VALUES
                {values_clause}
            ON CONFLICT (id) DO UPDATE
            SET pre_pt_supervoxel_id = EXCLUDED.pre_pt_supervoxel_id,
                post_pt_supervoxel_id = EXCLUDED.post_pt_supervoxel_id;
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


# Create the connection URL
DB_PASS = getpass(f"{DB_NAME} password: ")
connection_url = URL.create(
    drivername="postgresql+psycopg2",
    username=DB_USER,
    password=DB_PASS,
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME,
)


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
        # print(f'Loading new chunk: {chunk_bounds.bbox}...', end='', flush=True)
        assert seg_layer is not None
        assert seg_layer[chunk_bounds] is not None  # type: ignore[unreachable]
        chunk = seg_layer[chunk_bounds][0]
        # print('Chunk loaded.')
    relative_point = floor(seg_point - chunk_bounds.start)  # type: ignore[unreachable]
    return chunk[relative_point[0], relative_point[1], relative_point[2]]


def process_synapses(to_do):
    total_to_do = len(to_do)
    print(f"Looking up supervoxel IDs for {total_to_do} synapses...")
    done = []
    while to_do:
        # Process the top synapse.  Always fill in "A" if it wasn't
        # already known; and if we can fill in "B" at the same time
        # (usually the case), then it's done.
        synapse = to_do.pop()
        if synapse["segA"] is None:
            synapse["segA"] = lookup_segment_id(synapse["pointA"], True)
            if synapse["segB"] is None:
                synapse["segB"] = lookup_segment_id(synapse["pointB"], False)
        else:
            synapse["segB"] = lookup_segment_id(synapse["pointB"], True)
        if synapse["segB"] is None:
            to_do.append(synapse)
        else:
            done.append(synapse)

        # Now, see what other points we can fill in from the same chunk.
        for i in range(len(to_do) - 1, -1, -1):
            synapse = to_do[i]
            if synapse["segA"] is None:
                synapse["segA"] = lookup_segment_id(synapse["pointA"], False)
            if synapse["segB"] is None:
                synapse["segB"] = lookup_segment_id(synapse["pointB"], False)
            if synapse["segA"] is not None and synapse["segB"] is not None:
                del to_do[i]
                done.append(synapse)
        print(
            f"To-Do: {len(to_do)}/{total_to_do} "
            f"({round(100.0 - len(to_do)*100 / total_to_do)}% complete)"
            f"   Done: {len(done)}"
        )
        if len(done) > 500:
            print(f"Saving done records to {supervox_table}...")
            bulk_insert_seg_ids(done, supervox_table)
            done = []

    if done:
        print(f"Saving last done records to {supervox_table}...")
        bulk_insert_seg_ids(done, supervox_table)
    print(f"Finished processing {total_to_do} synapses!")


def main():
    global engine, synapse_table, supervox_table, synapse_resolution
    global seg_layer, seg_bounds, seg_resolution, seg_chunk_size, seg_offset
    global syn_to_seg

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
            "ipl_ribbon_synapses__dacey_human_fovea_2404"
        )
    synapse_table = supervox_table.split("__")[0]
    synapse_resolution = input_vec3Di("Synapse voxel scale (resolution)")

    print()
    seg_path = input("Segmentation volume path: ")
    data = read_gcs_json(seg_path + "/info")
    scale0 = data["scales"][0]
    seg_resolution = Vec3D(*scale0["resolution"])
    seg_chunk_size = Vec3D(*scale0["chunk_sizes"][0])
    seg_offset = Vec3D(*scale0["voxel_offset"])
    print(f"   Segmentation resolution: {seg_resolution}")
    print(f"   Segmentation chunk size: {seg_chunk_size}")
    print(f"   Segmentation offset:     {seg_offset}")

    syn_to_seg = synapse_resolution / seg_resolution
    print(f"   Conversion factor from synapse to segmentation: {syn_to_seg}")

    # Load the segmentation layer
    seg_layer, seg_bounds = load_volume(seg_path)

    to_do: List[dict] = []
    with engine.connect() as conn1:
        print("Reading synapses...")
        where_clause = f"""
NOT EXISTS (SELECT 1 FROM {supervox_table} b WHERE b.id = {synapse_table}.id);
"""
        to_do = read_synapses(conn1, synapse_table, where_clause)

    print(f"Read {len(to_do)} synapses from {synapse_table}")
    print("Sorting synapses into chunks...")
    bins = bin_synapses(to_do)
    print(f"Divided them into {len(bins)} bins")
    sorted_indexes = sorted(bins.keys())
    for i, bin_idx in enumerate(sorted_indexes):
        print()
        print(f"STARTING BIN {i}/{len(sorted_indexes)} (bin {bin_idx})...")
        process_synapses(bins[bin_idx])


if __name__ == "__main__":
    main()
