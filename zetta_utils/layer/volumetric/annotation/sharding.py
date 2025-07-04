"""
Code that supports writing precomputed annotation files in sharded format
(to greatly reduce the number of files on disk).  Reference:
https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/sharded.md
"""
# pylint: disable=too-many-branches,too-many-statements,too-many-locals

import gzip
import io
import os
import struct
from collections import defaultdict
from dataclasses import dataclass
from math import ceil
from typing import BinaryIO, Dict, List

from zetta_utils.layer.volumetric.annotation.annotations import ShardingSpec
from zetta_utils.layer.volumetric.annotation.utilities import path_join


@dataclass
class Chunk:
    """
    Represents a chunk of data to be stored in a shard.
    """

    chunk_id: int
    data: bytes


def get_shard_hex(shard_number: int, shard_bits: int) -> str:
    """Convert shard number to zero-padded lowercase hex string.

    :param shard_number: The shard number to convert
    :param shard_bits: Number of bits for the shard
    :return: Zero-padded lowercase hex string
    """
    padding = ceil(shard_bits / 4)
    return f"{shard_number:0{padding}x}"


def write_shard_file(
    output_file: BinaryIO, sharding_spec: ShardingSpec, shard_number: int, chunks: List[Chunk]
) -> None:
    """
    Write a shard file according to the Neuroglancer sharded format.

    :param output_file: File-like object to write the shard data to
    :param sharding_spec: The sharding specification
    :param shard_number: The shard number being written
    :param chunks: List of chunks that belong to this shard
    """
    # Group chunks by minishard
    minishard_chunks: Dict[int, List[Chunk]] = defaultdict(list)

    for chunk in chunks:
        expected_shard = sharding_spec.get_shard_number(chunk.chunk_id)
        if expected_shard != shard_number:
            raise ValueError(
                f"Chunk {chunk.chunk_id} belongs to shard {expected_shard}, "
                f"not shard {shard_number}"
            )

        minishard_num = sharding_spec.get_minishard_number(chunk.chunk_id)
        minishard_chunks[minishard_num].append(chunk)

    # Sort chunks within each minishard by chunk_id
    for minishard_num in minishard_chunks:
        minishard_chunks[minishard_num].sort(key=lambda c: c.chunk_id)

    # Build minishard indices and collect data
    num_minishards = sharding_spec.num_minishards_per_shard
    minishard_indices = {}
    minishard_data_sections = {}
    current_data_offset = 0

    # Keep track of where the next chunk of data is going to appear,
    # relative to the start of the minishard indexes.  The total
    # minishard index length is 24 (3 uint64's) times the number of chunks.
    next_data_pos = 24 * len(chunks)
    for minishard_num in range(num_minishards):
        chunks_in_minishard = minishard_chunks.get(minishard_num, [])

        if not chunks_in_minishard:
            # Empty minishard
            minishard_indices[minishard_num] = b""
            minishard_data_sections[minishard_num] = b""
            continue

        # Build minishard index arrays
        chunk_ids = []
        data_offsets = []
        data_sizes = []

        # Process chunks and apply data encoding
        # (ToDo: figure out if we're supposed to gzip each chunk separately like this,
        # which would pretty much never be a good idea for annotations, or gzip the
        # entire data section at once.  The spec is unclear on this.)
        encoded_chunk_data = []
        for chunk in chunks_in_minishard:
            if sharding_spec.data_encoding == "gzip":
                encoded_data = gzip.compress(chunk.data)
            else:  # "raw"
                encoded_data = chunk.data

            encoded_chunk_data.append(encoded_data)
            data_sizes.append(len(encoded_data))

        # Delta encode chunk IDs
        prev_id = 0
        for chunk in chunks_in_minishard:
            chunk_ids.append(chunk.chunk_id - prev_id)
            prev_id = chunk.chunk_id

        # Delta encode data offsets relative to end of previous chunk
        # (with the first one equal to the next data position).
        for i, data_size in enumerate(data_sizes):
            if i == 0:
                data_offsets.append(next_data_pos)
            else:
                # Subsequent chunks: 0 additional offset, as we're not
                # putting any extra space between chunks
                data_offsets.append(0)
            next_data_pos += data_size

        # Build the minishard index binary data
        # Format: [3, n] array of uint64le values
        # array[0, :] = delta-encoded chunk IDs
        # array[1, :] = delta-encoded data offsets (from end of prior chunk)
        # array[2, :] = data sizes
        index_data = io.BytesIO()

        # Write chunk IDs (delta encoded)
        for chunk_id_delta in chunk_ids:
            index_data.write(struct.pack("<Q", chunk_id_delta))

        # Write data offsets (delta encoded)
        for offset_delta in data_offsets:
            index_data.write(struct.pack("<Q", offset_delta))

        # Write data sizes
        for size in data_sizes:
            index_data.write(struct.pack("<Q", size))

        raw_index = index_data.getvalue()

        # Apply minishard index encoding
        # NOTE: doing this will make the initial data offset above wrong.
        # This is a circular dependency: we need to know the final size of
        # all minishard indexes, but that will change depending on what
        # it contains (including that offset).  I see no obvious way to
        # resolve that.  So, don't use gzip!
        if sharding_spec.minishard_index_encoding == "gzip":
            encoded_index = gzip.compress(raw_index)
        else:  # "raw"
            encoded_index = raw_index

        minishard_indices[minishard_num] = encoded_index

        # Concatenate all chunk data for this minishard
        minishard_data = b"".join(encoded_chunk_data)
        minishard_data_sections[minishard_num] = minishard_data

        current_data_offset += len(minishard_data)

    # Calculate minishard index positions
    minishard_index_offsets = {}
    current_index_offset = 0

    for minishard_num in range(num_minishards):
        start_offset = current_index_offset
        end_offset = start_offset + len(minishard_indices[minishard_num])
        minishard_index_offsets[minishard_num] = (start_offset, end_offset)
        current_index_offset = end_offset

    # Write shard index (2**minishard_bits * 16 bytes)
    for minishard_num in range(num_minishards):
        start_offset, end_offset = minishard_index_offsets[minishard_num]
        output_file.write(struct.pack("<Q", start_offset))  # start_offset: uint64le
        output_file.write(struct.pack("<Q", end_offset))  # end_offset: uint64le

    # Write minishard indices
    for minishard_num in range(num_minishards):
        output_file.write(minishard_indices[minishard_num])

    # Write chunk data
    for minishard_num in range(num_minishards):
        output_file.write(minishard_data_sections[minishard_num])


def write_shard_to_file(
    filepath: str, sharding_spec: ShardingSpec, shard_number: int, chunks: List[Chunk]
) -> None:
    """
    Convenience function to write a shard file to disk.

    :param filepath: Path where the shard file should be written
    :param sharding_spec: The sharding specification
    :param shard_number: The shard number being written
    :param chunks: List of chunks that belong to this shard
    """
    with open(os.path.expanduser(filepath), "wb") as f:
        write_shard_file(f, sharding_spec, shard_number, chunks)
        # print(f'Wrote {len(chunks)} items to shard file: {filepath}')


def write_shard_files(dir_path: str, sharding_spec: ShardingSpec, chunks: List[Chunk]) -> None:
    # Sort chunks into groups by shard number
    qty_shards = sharding_spec.num_shards
    shard_chunks: List[List[Chunk]] = list([] for _ in range(0, qty_shards))
    for chunk in chunks:
        shard_num = sharding_spec.get_shard_number(chunk.chunk_id)
        shard_chunks[shard_num].append(chunk)
    # Then, write 'em out!
    dir_path = os.path.expanduser(dir_path)
    os.makedirs(dir_path, exist_ok=True)
    for i in range(0, qty_shards):
        shard_hex = get_shard_hex(i, sharding_spec.shard_bits)
        file_path = path_join(dir_path, f"{shard_hex}.shard")
        write_shard_to_file(file_path, sharding_spec, i, shard_chunks[i])
