#!/usr/bin/env python
"""
This script takes lines from a precomputed annotations file,
and stuffs them into a synapses table in CAVE.
"""
import readline
import sys
from getpass import getpass
from typing import Dict, List

import nglui
from caveclient import CAVEclient
from sqlalchemy import create_engine
from sqlalchemy import text as sql
from sqlalchemy.engine import URL

from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric.annotation import build_annotation_layer

# Database connection parameters
DB_USER = "postgres"
DB_NAME = "dacey_human_fovea"
DB_HOST = "127.0.0.1"  # Local proxy address; run Cloud SQL Auth Proxy
DB_PORT = 5432  # Default PostgreSQL port

# Other parameters
POSITION_PRECISION = 0  # 0 = round to whole voxels, 1 = round to tenths, etc.

client = None
engine = None


def verify_cave_auth():
    # pylint: disable-next=global-statement
    global client
    client = CAVEclient()
    try:
        # pylint: disable-next=pointless-statement    # bite me, pylint
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


def get_annotation_layer_name(state):
    names = nglui.parser.annotation_layers(state)
    if len(names) == 0:
        print("No annotation layers found in this state.")
        sys.exit()
    elif len(names) == 1:
        return names[0]
    while True:
        for i, name in enumerate(names, start=1):
            print(f"{i}. {name}")
        choice = input("Enter layer name or number: ")
        if choice in names:
            return choice
        ichoice = int(choice) - 1
        if 0 <= ichoice < len(names):
            return names[ichoice]


def input_vec3D(prompt="", default=None):
    while True:
        s = input(prompt + (f" [{default.x}, {default.y}, {default.z}]" if default else "") + ": ")
        if s == "" and default:
            return default
        try:
            x, y, z = map(float, s.replace(",", " ").split())
            return Vec3D(x, y, z)
        # pylint: disable-next=bare-except
        except:
            print("Enter x, y, and z values separated by commas or spaces.")


def input_vec3Di(prompt="", default=None):
    v = input_vec3D(prompt, default)
    return round(v)


def bulk_insert_synapses(items: List[Dict], table_name: str, batch_size: int = 1000):
    """
    Bulk insert synapses into the database.

    Parameters:
        items: List of dictionaries, each containing 'pointA' and 'pointB' as (x,y,z) sequences
        batch_size: Number of rows to insert in each batch (default 1000)
    """

    def to_number(s):
        return round(float(s), POSITION_PRECISION)

    def create_batch_query(batch_items: List[Dict]) -> str:
        value_entries = []
        for item in batch_items:
            # Cast all coordinates to float for validation/safety
            pre_x, pre_y, pre_z = map(to_number, item["pointA"])
            post_x, post_y, post_z = map(to_number, item["pointB"])
            ctr_x = round((pre_x + post_x) / 2, POSITION_PRECISION)
            ctr_y = round((pre_y + post_y) / 2, POSITION_PRECISION)
            ctr_z = round((pre_z + post_z) / 2, POSITION_PRECISION)

            value_entries.append(
                f"""(
                CURRENT_TIMESTAMP, TRUE,
                ST_MakePoint({pre_x}, {pre_y}, {pre_z}),
                ST_MakePoint({post_x}, {post_y}, {post_z}),
                ST_MakePoint({ctr_x}, {ctr_y}, {ctr_z})
            )"""
            )

        values_clause = ",\n".join(value_entries)

        return f"""
            INSERT INTO {table_name}
                (created, valid, pre_pt_position, post_pt_position, ctr_pt_position)
            VALUES
                {values_clause}
        """

    assert engine is not None
    with engine.connect() as conn:  # type: ignore[unreachable]  # silly mypy
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            query = sql(create_batch_query(batch))
            conn.execute(query)
            print(f"Processed {min(i + batch_size, len(items))}/{len(items)} items")

        conn.commit()
        print(f"Committed {len(items)} new rows to {table_name}")


def load_annotations():
    layer = None
    items = None
    print("Enter Neuroglancer state link or ID, or a GS file path:")
    inp = input("> ")
    if inp.startswith("gs:"):
        layer = build_annotation_layer(inp, mode="read")
    else:
        verify_cave_auth()
        state_id = inp.split("/")[-1]  # in case full URL was given

        assert client is not None
        state = client.state.get_state_json(state_id)
        print("Select annotation layer containing synapses to import:")
        anno_layer_name = get_annotation_layer_name(state)
        data = nglui.parser.get_layer(state, anno_layer_name)

        if "annotations" in data:
            items = data["annotations"]
        elif "source" in data:
            print("Precomputed annotation layer.")
            layer = build_annotation_layer(data["source"], mode="read")
        else:
            print("Neither 'annotations' nor 'source' found in layer data.  I'm stumped.")
            sys.exit()
    if items is None and layer is not None:
        opt = ""
        while opt not in ("A", "B"):
            opt = input("Read [A]ll lines, or only within some [B]ounds? ").upper()
        if opt == "B":
            bbox_start = input_vec3Di("  Bounds start")
            bbox_end = input_vec3Di("    Bounds end")
            resolution = input_vec3Di("    Resolution")
            bbox = BBox3D.from_coords(bbox_start, bbox_end, resolution)
            lines = layer.backend.read_in_bounds(bbox, strict=True)
        else:
            lines = layer.backend.read_all()
        items = [
            {"id": hex(l.id)[2:], "type": "line", "pointA": l.start, "pointB": l.end}
            for l in lines
        ]
    return items


def main():
    password = getpass("Enter DB password: ")

    # Create the connection URL
    connection_url = URL.create(
        drivername="postgresql+psycopg2",
        username=DB_USER,
        password=password,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
    )

    # Create the engine
    # pylint: disable-next=global-statement
    global engine
    engine = create_engine(connection_url, connect_args={"connect_timeout": 10})

    # Try to connect, just to be sure we can
    try:
        print(f"Connecting to {DB_NAME} at {DB_HOST}:{DB_PORT}...")
        with engine.connect():
            print("Successfully connected to the database!")

    # pylint: disable-next=broad-exception-caught
    except Exception as e:
        print("Error connecting to the database:")
        print(e)
        sys.exit()

    items = load_annotations()

    print(f"{len(items)} annotations ready to export.")
    table_name = input("Table name: ")
    yn = input(f"CONFIRM: Import {len(items)} synapses into {table_name} [Y/n]? ").upper()
    if yn in ("Y", ""):
        bulk_insert_synapses(items, table_name, batch_size=1000)


if __name__ == "__main__":
    main()
