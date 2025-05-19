"""
This interactive script provides a variety of functions related to managing
synapse data in CAVE databases.
"""

import readline
import struct
import sys
from binascii import unhexlify
from getpass import getpass
from typing import Any, Dict, List, Tuple

import psycopg2
from caveclient import CAVEclient
from geoalchemy2.elements import WKBElement
from shapely.wkb import loads as load_wkb
from sqlalchemy import create_engine, inspect
from sqlalchemy import text as sql
from sqlalchemy.engine import URL, CursorResult, Result
from sqlalchemy.exc import SQLAlchemyError
from tabulate import tabulate

from zetta_utils.db_annotations.precomp_annotations import (
    AnnotationLayer,
    LineAnnotation,
)
from zetta_utils.geometry import Vec3D

# Global variables
DB_USER = "postgres"
DB_HOST = "127.0.0.1"  # Local proxy address; run Cloud SQL Auth Proxy
DB_PORT = 5432  # Default PostgreSQL port
db_name = ""  # datastack name, or 'NGState' if we only care about getting Neuroglancer states
password = ""
engine: Any = None  # sqlalchemy DB engine
cave_client: CAVEclient = None
synapse_table_name = ""


def input_vec3D(prompt="", default=None):
    while True:
        s = input(prompt + (f" [{default.x}, {default.y}, {default.z}]" if default else "") + ": ")
        if s == "" and default:
            return default
        try:
            x, y, z = map(float, s.replace(",", " ").split())
            return Vec3D(x, y, z)
        except Exception as err:
            print(err)
            print("Enter x, y, and z values separated by commas or spaces.")


def input_vec3Di(prompt="", default=None):
    v = input_vec3D(prompt, default)
    return round(v)


def input_yn(prompt="", default="y"):
    """
    Prompt the user for a Yes/No answer;
    returns True if yes, False if no.
    """
    default = default.lower()
    y = "Y" if default == "y" else "y"
    n = "N" if default == "n" else "n"
    while True:
        yn = input(f"{prompt} [{y}/{n}]? ").lower()
        if yn == "":
            yn = default
        if yn == "y":
            return True
        if yn == "n":
            return False


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


def format_db_value(value: Any, column_name: str) -> str:
    """Format a value based on column type and name."""
    if value is None:
        return "NULL"
    if isinstance(value, WKBElement) or column_name.endswith("_position"):
        try:
            if isinstance(value, str) and all(c in "0123456789ABCDEFabcdef" for c in value):
                x, y, z = parse_wkb_hex_point(value)
                return f"{x}, {y}, {z}"
            elif hasattr(value, "x") and hasattr(value, "y"):
                return f"{value.x}, {value.y}" + (f", {value.z}" if hasattr(value, "z") else "")
            elif isinstance(value, WKBElement):
                shape = load_wkb(bytes(value.data))
                return f"{shape.x}, {shape.y}" + (f", {shape.z}" if hasattr(shape, "z") else "")
        # pylint: disable-next=broad-exception-caught
        except Exception:
            print("GOT EXCEPTION IN WKB")
            return str(value)
    return str(value)


def print_db_results(result: Result, batch_size: int = 20) -> None:
    """Print query results in a paginated table format."""
    try:
        if result.returns_rows:  # type: ignore[attr-defined]
            rows = result.fetchall()
            if not rows:
                print("Query returned no results.")
                return

            headers = list(result.keys())
            formatted_rows = [
                [format_db_value(value, col) for value, col in zip(row, headers)] for row in rows
            ]

            total_rows = len(rows)
            start_idx = 0

            while start_idx < total_rows:
                batch = formatted_rows[start_idx : start_idx + batch_size]
                print(tabulate(batch, headers=headers, tablefmt="psql"))
                start_idx += batch_size

                if start_idx < total_rows:
                    choice = input(
                        f"\nShowing {start_idx}/{total_rows} rows. "
                        "Press Enter for more, R for Rest, Q to quit: "
                    ).lower()
                    if choice == "q":
                        break
                    if choice == "r":
                        batch_size = len(formatted_rows) - start_idx
                else:
                    print(f"\nTotal rows: {total_rows}")
        elif isinstance(result, CursorResult):
            # For INSERT, UPDATE, DELETE statements
            print(f"Success. Rows affected: {result.rowcount}")
        else:
            # For other kinds of results (?)
            print("Success.")

    except SQLAlchemyError as e:
        print(f"Error executing query: {str(e)}")


def get_db_password(force_ask: bool = False):
    # ToDo: store passwords in a local file, and try them before prompting
    # the user.
    global password
    if force_ask or not password:
        password = getpass(f"Password for DB service at {DB_HOST}: ")
    return password != ""


def select_database() -> bool:
    global db_name
    if not get_db_password():
        return False
    try:
        conn = psycopg2.connect(
            host="127.0.0.1",
            port="5432",
            database="postgres",  # Default database that always exists
            user="postgres",
            password=password,
        )
    except Exception as err:
        print("Unable to connect to database.")
        print(err)
        return False

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
            choice = input("Enter DB name or number: ")
        except EOFError:
            print()
            return False
        if choice in databases:
            db_name = choice
            return True
        choice_i = int(choice)
        if choice_i in num_to_db:
            db_name = num_to_db[choice_i]
            return True


def ensure_engine() -> bool:
    global engine, db_name
    if engine:
        return True
    select_database()
    if not db_name:
        return False

    connection_url = URL.create(
        drivername="postgresql+psycopg2",
        username=DB_USER,
        password=password,
        host=DB_HOST,
        port=DB_PORT,
        database=db_name,
    )
    engine = create_engine(connection_url, connect_args={"connect_timeout": 10})

    # Try to connect, just to be sure we can
    try:
        print(f"Connecting to {db_name} at {DB_HOST}:{DB_PORT}...")
        with engine.connect() as connection:
            print("Successfully connected.")
            return True
    # pylint: disable-next=broad-exception-caught
    except Exception as e:
        print("Error connecting to the database:")
        print(e)
        return False


def verify_cave_auth() -> bool:
    global cave_client, db_name

    if not db_name:
        db_name = input("Enter data stack (DB) name: ")

    # ToDo: figure out a more organized/sensible way to figure out the right
    # server address and auth token for each database.  For now:
    # ToDo: figure out a more organized/sensible way to figure out the right
    # server address and auth token for each database.  For now:
    if db_name == "NGState" or db_name.startswith("wclee"):
        cave_client = CAVEclient(
            datastack_name=db_name if db_name != "NGState" else None,
            server_address="https://global.daf-apis.com",
        )
    else:
        cave_client = CAVEclient(
            datastack_name=db_name, server_address="https://proofreading.zetta.ai"
        )

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


def do_db_command(command, connection, pending_commit=False) -> Any:
    """
    Execute the given SQL command, or one of our extra commands
    (commit, or rollback).  Return the new value for
    pending_commit, indicating changes are pending.
    """
    if command.lower() == "commit":
        connection.commit()
        print("Changes commited.")
        pending_commit = False
    elif command.lower() == "rollback":
        connection.rollback()
        print("Changes rolled back.")
        pending_commit = False
    else:
        # SQL (possibly multi-line) command
        while not command.endswith(";"):
            try:
                command += " " + input("...>")
            except EOFError:
                command = ""
                break
        if not command:
            return pending_commit
        try:
            result = connection.execute(sql(command))
            print_db_results(result)
            if not result.returns_rows and result.rowcount > 0:
                pending_commit = True
        # pylint: disable-next=broad-exception-caught
        except Exception as e:
            print(e)
    return pending_commit


def list_tables(header="Tables:", include_views=False):
    print(header)
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    if include_views:
        table_names += inspector.get_view_names()
    table_names.sort()
    print(" - " + "\n - ".join(table_names))
    print()
    return table_names


def create_synapse_table():
    global synapse_table_name
    verify_cave_auth()
    synapse_table_name = input("Table name:  ")
    resolution = list(input_vec3Di("Resolution:  "))
    desc = input("Description: ")
    print()
    if input_yn(f"CONFIRM: Create table '{synapse_table_name}' with resolution {resolution}"):
        cave_client.annotation.create_table(
            table_name=synapse_table_name,
            schema_name="synapse",
            voxel_resolution=resolution,
            description=desc,
        )
        print(f"Created: {synapse_table_name}")


def create_supervoxel_table():
    global synapse_table_name, password
    if not synapse_table_name:
        synapse_table_name = input("Base table name: ")
    pcg_id = input("PyChunkedGraph ID: ")
    assert pcg_id
    table_name = f"{synapse_table_name}__{pcg_id}"
    if not input_yn(f"CONFIRM: Create table '{table_name}'"):
        return
    connection_url = URL.create(
        drivername="postgresql+psycopg2",
        username=DB_USER,
        password=password,
        host=DB_HOST,
        port=DB_PORT,
        database=db_name,
    )
    engine = create_engine(
        connection_url, connect_args={"connect_timeout": 10}
    )  # 10 seconds timeout
    with engine.connect() as connection:
        command = f"""
CREATE TABLE public.{table_name} (
	id bigint NOT NULL,
	pre_pt_supervoxel_id bigint,
	pre_pt_root_id bigint,
	post_pt_supervoxel_id bigint,
	post_pt_root_id bigint
);
ALTER TABLE public.{table_name} OWNER TO postgres;
ALTER TABLE ONLY public.{table_name}
	ADD CONSTRAINT {table_name}_pkey PRIMARY KEY (id);
"""
        connection.execute(sql(command))
        connection.commit()
        print(f"Created: {table_name}")

        # We also need to add an entry describing this table
        # in the segmentation_table_metadata table.
        command = f"""
INSERT INTO segmentation_table_metadata
    (schema_type, table_name, valid, created, pcg_table_name, annotation_table)
    VALUES (
        'synapse',
        '{table_name}',
        TRUE,
        NOW(),
        '{pcg_id}',
        '{synapse_table_name}'
);"""
        connection.execute(sql(command))
        connection.commit()
        print(f"Inserted new record into segmentation_table_metadata.")


def sql_main_loop():
    if not ensure_engine():
        return

    # List the tables.  This is the main thing the user needs to know in order to
    # get anything done, and is most likely to have forgotten.
    list_tables("Available Tables and Views:", True)

    # Now, do the SQL thing.
    with engine.connect() as connection:
        pending_commit = False
        while True:
            try:
                inp = input("SQL or COMMIT> " if pending_commit else "SQL> ").strip()
                if not inp:
                    continue
            except EOFError:
                print("\nExiting.")
                return
            if inp.lower().strip(" \n\t;") in ("quit", "exit", "menu"):
                return
            pending_commit = do_db_command(inp, connection, pending_commit)


def create_tables():
    if not ensure_engine():
        return
    if not verify_cave_auth():
        return

    list_tables("Current Tables:")
    print()
    if input_yn("CREATE NEW SYNAPSE TABLE"):
        create_synapse_table()

    print()
    if input_yn("CREATE RELATED SUPERVOXEL TABLE"):
        result = create_supervoxel_table()


def create_index(table_name: str, index_name: str, using_clause: str):
    with engine.connect() as connection:
        check = f"SELECT indexname FROM pg_indexes WHERE tablename='{table_name}' and indexname='{index_name}'"
        if connection.scalar(sql(check)) is not None:
            print(f"{index_name}: already exists.")
        else:
            index_def = f"CREATE INDEX {index_name} ON public.{table_name} USING {using_clause};"
            connection.execute(sql(index_def))
            print(f"{index_name}: created.")


def create_indexes():
    if not ensure_engine():
        return
    names = list_tables("Tables:")
    while True:
        table_name = input("Enter name of table to index: ")
        if table_name in names:
            break
        if not table_name or table_name in ["exit", "quit", "menu"]:
            return
        print('Invalid table name.  Press Return or enter "menu" to return to the main menu.')
    print()
    with engine.connect() as connection:
        print("Current indexes:")
        cmd = f"SELECT indexname, indexdef FROM pg_indexes WHERE tablename='{table_name}';"
        do_db_command(cmd, connection)

    if not input_yn(f"CONFIRM: create indexes for {table_name}"):
        return

    t = table_name
    if "__" in table_name:
        create_index(table_name, f"{t}_pkey", "btree (id)")
        create_index(table_name, f"{t}_pre_pt_root_id", "btree (pre_pt_root_id)")
        create_index(table_name, f"{t}_post_pt_root_id", "btree (post_pt_root_id)")
    else:
        create_index(table_name, f"{t}_pkey", "btree (id)")
        create_index(
            table_name, f"idx_{t}_pre_pt_position", "gist (pre_pt_position gist_geometry_ops_nd)"
        )
        create_index(table_name, f"ix_{t}_deleted", "btree (deleted)")
        create_index(
            table_name, f"idx_{t}_ctr_pt_position", "gist (ctr_pt_position gist_geometry_ops_nd)"
        )
        create_index(table_name, f"ix_{t}_created", "USING btree (created)")
        create_index(
            table_name,
            f"idx_{t}_post_pt_position",
            "USING gist (post_pt_position gist_geometry_ops_nd)",
        )
    input("\n(Press Return.)")


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


def export_to_annotations():
    # List the tables, and get the name of the synapse table.
    ensure_engine()
    list_tables()
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


def print_title():
    print(
        """\n\n\n\n
        ______   _____   __    ___  ______
       /  ___/  /     | |  |  /  / /  ___/
      /  /     /  /|  | |  | /  / /  /_
     /  /     /  /_|  | |  |/  / /  __/
    /  /___  /  ___   | |     / /  /___
   /______/ /__/   |__| |____/ /______/
      S Y N A P S E   M A N A G E R
\n\n"""
    )


def main_menu():
    while True:
        print_title()
        options = [
            ("SQL client", sql_main_loop),
            ("Create synapse table(s)", create_tables),
            ("Add indexes to synapse table", create_indexes),
            ("Export from CAVE to precomputed annotations", export_to_annotations),
        ]
        for i, opt in enumerate(options):
            print(f"   {i+1}. {opt[0]}")
        print("\n   Q. Quit\n")
        print
        while True:
            try:
                choice = input("=> ").upper()
            except EOFError:
                print()
                return
            if choice == "Q":
                return
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                options[idx][1]()
                break


if __name__ == "__main__":
    main_menu()
