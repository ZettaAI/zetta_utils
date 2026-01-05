"""
Helper utilities used by various code in this (volumetric.annotation) module.
"""

import os

from cloudfiles import CloudFile


def is_local_filesystem(path: str) -> bool:
    return path.startswith("file://") or "://" not in path


def path_join(*paths: str):
    if not paths:
        raise ValueError("At least one path is required")

    if not is_local_filesystem(paths[0]):  # pragma: no cover
        # Join paths using "/" for GCS or other URL-like paths
        cleaned_paths = [path.strip("/") for path in paths]
        return "/".join(cleaned_paths)
    else:
        # Use os.path.join for local file paths
        return os.path.join(*paths)


def write_bytes(file_or_gs_path: str, data: bytes):
    """
    Write bytes to a local file or Google Cloud Storage.

    :param file_or_gs_path: path to file to write (local or GCS path)
    :param data: bytes to write
    """
    if "//" not in file_or_gs_path:
        file_or_gs_path = "file://" + file_or_gs_path
    cf = CloudFile(file_or_gs_path)
    cf.put(data, cache_control="no-cache, no-store, max-age=0, must-revalidate")


def compressed_morton_code(cell_index, grid_shape):
    """Compute the compressed morton code of a cell index (used as the chunk id
    for sharding spatial chunks).
    Reference:
    https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/volume.md#compressed-morton-code
    """
    code = 0
    j = 0
    max_bits = max(grid_shape).bit_length()

    for i in range(max_bits):
        for dim in range(3):  # 0: x, 1: y, 2: z
            if (1 << i) < grid_shape[dim]:
                bit = (cell_index[dim] >> i) & 1
                code |= bit << j
                j += 1
    return code
