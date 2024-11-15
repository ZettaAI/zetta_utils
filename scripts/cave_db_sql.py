#!/usr/bin/env python
"""
This script provides a raw SQL interface to a CAVE database.
Use with caution!
"""
import readline
import struct
import sys
from binascii import unhexlify
from typing import Any

import psycopg2
from geoalchemy2.elements import WKBElement
from shapely.wkb import loads as load_wkb
from sqlalchemy import create_engine, inspect
from sqlalchemy import text as sql
from sqlalchemy.engine import URL, CursorResult, Result
from sqlalchemy.exc import SQLAlchemyError
from tabulate import tabulate

# Database connection parameters
DB_USER = "postgres"
DB_PASS = ""  # put DB password here
DB_HOST = "127.0.0.1"  # Local proxy address; run Cloud SQL Auth Proxy
DB_PORT = 5432  # Default PostgreSQL port


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


def format_value(value: Any, column_name: str) -> str:
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


def print_results(result: Result, batch_size: int = 20) -> None:
    """Print query results in a paginated table format."""
    try:
        if result.returns_rows:  # type: ignore
            rows = result.fetchall()
            if not rows:
                print("Query returned no results.")
                return

            headers = list(result.keys())
            formatted_rows = [
                [format_value(value, col) for value, col in zip(row, headers)] for row in rows
            ]

            total_rows = len(rows)
            start_idx = 0

            while start_idx < total_rows:
                batch = formatted_rows[start_idx : start_idx + batch_size]
                print(tabulate(batch, headers=headers, tablefmt="psql"))
                start_idx += batch_size

                if start_idx < total_rows:
                    if (
                        input(
                            f"\nShowing {start_idx}/{total_rows} rows. "
                            "Press Enter for more, q to quit: "
                        ).lower()
                        == "q"
                    ):
                        break
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
            choice: int | str = input("Enter DB name or number: ")
        except EOFError:
            print()
            return None
        if choice in databases:
            return choice
        choice = int(choice)
        if choice in num_to_db:
            return num_to_db[choice]


def do_command(command, connection, pending_commit=False) -> Any:
    """
    Execute the given SQL command, or one of our extra commands
    (quit, commit, or rollback).  Return the new value for
    pending_commit, indicating changes are pending.
    """
    if command.lower() in ("quit", "exit"):
        sys.exit()
    elif command.lower() == "commit":
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
            print_results(result)
            if not result.returns_rows and result.rowcount > 0:
                pending_commit = True
        # pylint: disable-next=broad-exception-caught
        except Exception as e:
            print(e)
    return pending_commit


def main():

    db_name = select_database()
    if db_name is None:
        return

    # Create the connection URL
    connection_url = URL.create(
        drivername="postgresql+psycopg2",
        username=DB_USER,
        password=DB_PASS,
        host=DB_HOST,
        port=DB_PORT,
        database=db_name,
    )

    # Create the engine
    engine = create_engine(
        connection_url, connect_args={"connect_timeout": 10}
    )  # 10 seconds timeout

    # Try to connect, just to be sure we can
    try:
        print(f"Connecting to {db_name} at {DB_HOST}:{DB_PORT}...")
        with engine.connect() as connection:
            print("Successfully connected.")
    # pylint: disable-next=broad-exception-caught
    except Exception as e:
        print("Error connecting to the database:")
        print(e)
        sys.exit()

    # List the tables.  This is the main thing the user needs to know in order to
    # get anything done, and is most likely to have forgotten.
    inspector = inspect(engine)
    print("Available tables:\n - " + "\n - ".join(inspector.get_table_names()))
    print()

    # Now, do the SQL thing.
    with engine.connect() as connection:
        pending_commit = False
        while True:
            try:
                inp = input("SQL or COMMIT>" if pending_commit else "SQL>").strip()
                if not inp:
                    continue
            except EOFError:
                print("\nExiting.")
                sys.exit()
            pending_commit = do_command(inp, connection, pending_commit)


if __name__ == "__main__":
    main()
