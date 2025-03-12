"""
This script exports synapses from a CAVE table to a precomputed-annotations
file stored in Google Cloud Storage.

You are responsible (at least for now) for ensuring that the DB table and
the precomputed file are using the same resolution.
"""

import argparse
import json
import sys
from collections import defaultdict
from math import floor
from typing import Dict, List, Tuple

import readline
import psycopg2
from google.cloud import storage
from sqlalchemy import create_engine, inspect
from sqlalchemy import text as sql
from sqlalchemy.engine import URL

from zetta_utils.db_annotations.precomp_annotations import (
    AnnotationLayer,
    LineAnnotation,
)
from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex
from zetta_utils.layer.volumetric.cloudvol import build_cv_layer
from zetta_utils.layer.volumetric.precomputed import PrecomputedInfoSpec

# Database connection parameters
DB_USER = "postgres"
DB_PASS = "Abracadabra is the passphrase"
DB_HOST = "127.0.0.1"  # Local proxy address; run Cloud SQL Auth Proxy
DB_PORT = 5432  # Default PostgreSQL port

# Globals
engine = None
synapse_table = ""  # e.g. ipl_ribbon_synapses
synapse_resolution = Vec3D(0, 0, 0)


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


def row_to_line(row: Tuple, keys: Tuple[str, ...]) -> LineAnnotation:
    """
    Convert a database row containing synapse data into a LineAnnotation.
    """
    row_dict = dict(zip(keys, row))
    return LineAnnotation(
        line_id=row_dict["id"],
        start=(float(row_dict["pre_x"]), float(row_dict["pre_y"]), float(row_dict["pre_z"])),
        end=(float(row_dict["post_x"]), float(row_dict["post_y"]), float(row_dict["post_z"])),
    )


def read_synapses(conn, table_name, where_clause) -> List[LineAnnotation]:
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
        result.append(row_to_line(rec, keys))
    return result


def input_vec3D(prompt="", default=None):
    while True:
        s = input(prompt + (f" [{default.x}, {default.y}, {default.z}]" if default else "") + ": ")
        if s == "" and default:
            return default
        try:
            x, y, z = map(float, s.replace(",", " ").split())
            return Vec3D(x, y, z)
        except:
            print("Enter x, y, and z values separated by commas or spaces.")


def input_vec3Di(prompt="", default=None):
    v = input_vec3D(prompt, default)
    return round(v)


def select_database():
    conn = psycopg2.connect(
        host="127.0.0.1",
        port="5432",
        database="postgres",  # Default database that always exists
        user="postgres",
        password="Abracadabra is the passphrase",
    )
    cur = conn.cursor()
    cur.execute("SELECT datname FROM pg_database")
    databases = [db[0] for db in cur.fetchall()]
    databases.sort()
    num_to_db = {}
    next_num = 1
    print("Available databases:")
    for db in databases:
        print(f"   {' ' if next_num < 10 else ''}{next_num}. {db}")
        num_to_db[next_num] = db
        next_num += 1
    print()
    while True:
        try:
            choice: str | int = input("Enter DB name or number: ")
        except EOFError:
            print()
            return None
        if choice in databases:
            return choice
        choice = int(choice)
        if choice in num_to_db:
            return num_to_db[choice]


def main():
    global engine, synapse_table

    db_name = select_database()

    # Create the engine
    connection_url = URL.create(
        drivername="postgresql+psycopg2",
        username=DB_USER,
        password=DB_PASS,
        host=DB_HOST,
        port=DB_PORT,
        database=db_name,
    )
    engine = create_engine(
        connection_url, connect_args={"connect_timeout": 10}
    )  # 10 seconds timeout

    # Try to connect, just to be sure we can
    try:
        print(f"Connecting to {db_name} at {DB_HOST}:{DB_PORT}...")
        with engine.connect() as connection:
            print("Successfully connected to the database!")

    # pylint: disable-next=broad-exception-caught
    except Exception as e:
        print("Error connecting to the database:")
        print(e)
        sys.exit()

    # List the tables, and get the name of the synapse table.
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    print("Tables:\n - " + "\n - ".join(table_names))
    print()
    synapse_table = input("Synapse table to export: ")

    print("Enter WHERE clause to restrict synapses, if desired:")
    where_clause = input("> ")
    if where_clause.upper().startswith("WHERE "):
        where_clause = where_clause[6:]
    with engine.connect() as conn1:
        print("Reading synapses...")
        to_do = read_synapses(conn1, synapse_table, where_clause)

    print(f"Read {len(to_do)} synapses from {synapse_table}")

    print()
    file_path = input("Export to path: ")
    layer = AnnotationLayer(file_path)
    if not layer.exists():
        print(f"Precomputed annotation file {file_path} does not exist.")
        print("Alas, this script does not yet have the capability to create one for you.")
        print("Please copy or otherwise create your target file first, then try again.")
        sys.exit()
    while True:
        opt = input("[C]lear existing data, or [A]dd to it? ").upper()
        if opt in ["C", "A"]:
            break
    if opt == "C":
        layer.clear()

    print(f"Writing {len(to_do)} lines to {file_path}...")
    layer.write_annotations(to_do, all_levels=False)
    print("Post-processing...")
    layer.post_process()
    print("All done!")


if __name__ == "__main__":
    main()
