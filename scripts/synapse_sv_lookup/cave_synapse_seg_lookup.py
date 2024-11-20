"""
This script looks up the segment (supervoxel) IDs for the endpoints of synapses
which are not already in the supervoxel table, and stuffs those ids into the
supervoxel table, in a CAVE annotation DB.

Command-line options:
    --dry-run: do everything except actually write to the DB
"""
import argparse
import sys
from collections import defaultdict
from math import floor
from typing import Dict, List, Tuple

from sqlalchemy import create_engine
from sqlalchemy import text as sql
from sqlalchemy.engine import URL

from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex
from zetta_utils.layer.volumetric.cloudvol import build_cv_layer
from zetta_utils.layer.volumetric.precomputed import PrecomputedInfoSpec

# Database connection parameters
DB_USER = "postgres"
DB_PASS = "Abracadabra is the passphrase"
DB_NAME = "dacey_human_fovea"
DB_HOST = "127.0.0.1"  # Local proxy address; run Cloud SQL Auth Proxy
DB_PORT = 5432  # Default PostgreSQL port

# Synapse table parameters
SYNAPSE_TABLE = "ipl_inhib_synapses"
SUPERVOX_TABLE = "ipl_inhib_synapses__dacey_human_fovea_2404"
SYNAPSE_RESOLUTION = Vec3D(20, 20, 50)

# Segmentation parameters
# pylint: disable-next=line-too-long
SEG_PATH = "gs://zetta_ws/dacey_human_fovea_2404"
SEG_RESOLUTION = Vec3D(40, 40, 50)
SEG_CHUNK_SIZE = Vec3D(256, 256, 64)
SEG_OFFSET = Vec3D(0, 512, 1)

# Conversion factors for going from one resolution to another
SYN_TO_SEG = SYNAPSE_RESOLUTION / SEG_RESOLUTION
SEG_TO_SYN = SEG_RESOLUTION / SYNAPSE_RESOLUTION

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()
if args.dry_run:
    print("Dry Run mode active.")


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
    offset_point = seg_point - SEG_OFFSET
    chunk_count = floor(offset_point / SEG_CHUNK_SIZE)
    chunk_start = SEG_OFFSET + chunk_count * SEG_CHUNK_SIZE
    return VolumetricIndex.from_coords(chunk_start, chunk_start + SEG_CHUNK_SIZE, SEG_RESOLUTION)


def bin_index_for_point(seg_point):
    return floor((seg_point - SEG_OFFSET[0]) / SEG_CHUNK_SIZE)


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
    return {
        "id": row_dict["id"],
        "pointA": Vec3D(
            float(row_dict["pre_x"]), float(row_dict["pre_y"]), float(row_dict["pre_z"])
        )
        * SYN_TO_SEG,
        "pointB": Vec3D(
            float(row_dict["post_x"]), float(row_dict["post_y"]), float(row_dict["post_z"])
        )
        * SYN_TO_SEG,
        "segA": None,
        "segB": None,
    }


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

    with engine.connect() as conn:
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
engine = create_engine(connection_url, connect_args={"connect_timeout": 10})  # 10 seconds timeout

# Try to connect, just to be sure we can
try:
    print(f"Connecting to {DB_NAME} at {DB_HOST}:{DB_PORT}...")
    with engine.connect() as connection:
        print("Successfully connected to the database!")

# pylint: disable-next=broad-exception-caught
except Exception as e:
    print("Error connecting to the database:")
    print(e)
    sys.exit()

# Load the segmentation layer
seg_layer, seg_bounds = load_volume(SEG_PATH)

chunk = None
chunk_bounds = None


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
        chunk = seg_layer[chunk_bounds][0]
        # print('Chunk loaded.')
    relative_point = floor(seg_point - chunk_bounds.start)
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
            print(f"Saving done records to {SUPERVOX_TABLE}...")
            bulk_insert_seg_ids(done, SUPERVOX_TABLE)
            done = []

    if done:
        print(f"Saving last done records to {SUPERVOX_TABLE}...")
        bulk_insert_seg_ids(done, SUPERVOX_TABLE)
    print(f"Finished processing {total_to_do} synapses!")


def main():
    to_do: List[dict] = []
    with engine.connect() as conn1:
        print("Reading synapses...")
        where_clause = f"""
NOT EXISTS (SELECT 1 FROM {SUPERVOX_TABLE} b WHERE b.id = {SYNAPSE_TABLE}.id);
"""
        where_clause = ""  # HACK!!!
        to_do = read_synapses(conn1, SYNAPSE_TABLE, where_clause)

    print(f"Read {len(to_do)} synapses from {SYNAPSE_TABLE}")
    print(f"Sorting synapses into chunks...")
    bins = bin_synapses(to_do)
    print(f"Divided them into {len(bins)} bins")
    sorted_indexes = sorted(bins.keys())
    for i in range(0, len(sorted_indexes)):
        print()
        bin_idx = sorted_indexes[i]
        print(f"STARTING BIN {i}/{len(sorted_indexes)} (bin {bin_idx})...")
        process_synapses(bins[bin_idx])


if __name__ == "__main__":
    main()
