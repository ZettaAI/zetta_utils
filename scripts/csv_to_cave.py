"""
Reads a CSV file containing full synapse data (positions and supervoxel IDs),
and writes this data to CAVE (base table and supervox table).
"""
from datetime import datetime
from io import StringIO
from typing import Any

import readline
import pandas as pd
from cloudfiles import CloudFile
from geoalchemy2 import Geometry
from sqlalchemy import Boolean, Column, DateTime, Integer, create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    declarative_base,
    mapped_column,
    sessionmaker,
)

# Database connection parameters
DB_USER = "postgres"
db_pass = ''
DB_HOST = "127.0.0.1"  # Local proxy address; run Cloud SQL Auth Proxy
DB_PORT = 5432  # Default PostgreSQL port


class Base(DeclarativeBase):
    pass


def create_synapse_tables(synapse_table: str, aux_table: str):
    class Synapse(Base):
        __tablename__ = synapse_table

        id: Mapped[int] = mapped_column(Integer, primary_key=True)
        ctr_pt_position: Mapped[Any] = mapped_column(Geometry("POINT Z"))
        pre_pt_position: Mapped[Any] = mapped_column(Geometry("POINT Z"))
        post_pt_position: Mapped[Any] = mapped_column(Geometry("POINT Z"))
        valid: Mapped[bool] = mapped_column(Boolean)
        created: Mapped[datetime] = mapped_column(DateTime)

    class SynapseAux(Base):
        __tablename__ = aux_table

        id: Mapped[int] = mapped_column(Integer, primary_key=True)
        pre_pt_supervoxel_id: Mapped[int] = mapped_column(Integer)
        post_pt_supervoxel_id: Mapped[int] = mapped_column(Integer)
        pre_pt_root_id: Mapped[int] = mapped_column(Integer)
        post_pt_root_id: Mapped[int] = mapped_column(Integer)

    return Synapse, SynapseAux


def point_to_wkt(x, y, z):
    return f"POINT Z({x} {y} {z})"


class CSVChunkReader:
    def __init__(self, cloud_path, chunk_size_bytes=1024 * 1024):
        self.cf = CloudFile(cloud_path)
        self.chunk_size = chunk_size_bytes
        self.file_size = self.cf.size()
        self.pos = 0
        self.buffer = ""

        # Read and store header separately
        header_chunk = self.cf[0:1024].decode("utf-8")  # Assume header < 1KB
        first_newline = header_chunk.index("\n")
        self.header = header_chunk[:first_newline].strip()
        self.pos = first_newline + 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.pos >= self.file_size and not self.buffer:
            raise StopIteration

        while True:
            if self.pos < self.file_size:
                chunk = self.cf[self.pos : self.pos + self.chunk_size].decode("utf-8")
                self.pos += len(chunk)

                last_newline = chunk.rfind("\n")
                if last_newline != -1:
                    # Add complete lines to buffer
                    process_chunk = self.buffer + chunk[: last_newline + 1]
                    # Save partial line for next time
                    self.buffer = chunk[last_newline + 1 :]

                    if process_chunk:
                        return pd.read_csv(StringIO(self.header + "\n" + process_chunk))
                else:
                    self.buffer += chunk
            else:
                # Process any remaining data
                if self.buffer:
                    df = pd.read_csv(StringIO(self.header + "\n" + self.buffer))
                    self.buffer = ""
                    return df
                raise StopIteration


def process_batch(engine, session, df_batch, Synapse, SynapseAux):
    # First, insert all synapses and get their IDs
    stmt = Synapse.__table__.insert().returning(Synapse.__table__.c.id)
    synapse_values = [
        {
            "ctr_pt_position": point_to_wkt(
                round(row["centroid_x"]), round(row["centroid_y"]), round(row["centroid_z"])
            ),
            "pre_pt_position": point_to_wkt(
                round(row["presyn_x"]), round(row["presyn_y"]), round(row["presyn_z"])
            ),
            "post_pt_position": point_to_wkt(
                round(row["postsyn_x"]), round(row["postsyn_y"]), round(row["postsyn_z"])
            ),
            "valid": True,
            "created": datetime.now(),
        }
        for _, row in df_batch.iterrows()
    ]
    result = session.execute(stmt, synapse_values)
    synapse_ids = [row[0] for row in result]

    # Then create aux records with the correct IDs
    aux_values = [
        {
            "id": synapse_id,
            "pre_pt_supervoxel_id": row.presyn_segid,
            "post_pt_supervoxel_id": row.postsyn_segid,
            "pre_pt_root_id": None,
            "post_pt_root_id": None,
        }
        for synapse_id, row in zip(synapse_ids, df_batch.itertuples())
    ]
    session.execute(SynapseAux.__table__.insert(), aux_values)
    session.commit()

    return len(synapse_ids)


def main():
    csv_path = input("Enter CSV file path: ")
    db_name = input("Enter database name: ")
    db_pass = input("Enter DB password: ")
    synapse_table = input("Enter synapse table name: ")
    pcg_id = input("Enter PyChunkedGraph ID: ")

    aux_table = f"{synapse_table}__{pcg_id}"

    Synapse, SynapseAux = create_synapse_tables(synapse_table, aux_table)

    connection_url = URL.create(
        drivername="postgresql+psycopg2",
        username=DB_USER,
        password=db_pass,
        host=DB_HOST,
        port=DB_PORT,
        database=db_name,
    )
    engine = create_engine(connection_url, connect_args={"connect_timeout": 10})
    Session = sessionmaker(bind=engine)
    session = Session()

    total_rows = 0
    chunk_reader = CSVChunkReader(csv_path)

    for chunk_num, df_chunk in enumerate(chunk_reader):
        rows_processed = process_batch(engine, session, df_chunk, Synapse, SynapseAux)
        total_rows += rows_processed
        pct_done = round(chunk_reader.pos / chunk_reader.file_size * 100, 2)
        print(
            f"Processed batch {chunk_num + 1}: {rows_processed} records "
            f"(Total: {total_rows}); {pct_done}% complete"
        )

    print(f"\nCompleted! Total records processed: {total_rows}")


if __name__ == "__main__":
    main()
