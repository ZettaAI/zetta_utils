from __future__ import annotations

import math
from typing import Literal, Sequence

from zetta_utils import builder
from zetta_utils.geometry import BBox3D, IntVec3D, Vec3D
from zetta_utils.layer.volumetric.tabular.backend import TabularBackend, read_info
from zetta_utils.layer.volumetric.tabular.layer import VolumetricTabularLayer


def _resolve_bounds(
    bbox: BBox3D | None,
    voxel_offset: Sequence[int] | None,
    dataset_size: Sequence[int] | None,
    res: Vec3D,
) -> tuple[IntVec3D, IntVec3D]:
    """Convert bbox or voxel_offset+dataset_size into (voxel_offset, size) Vec3Ds."""
    if bbox is not None:
        if voxel_offset is not None or dataset_size is not None:
            raise ValueError(
                "When `bbox` is provided, `voxel_offset` and `dataset_size` should not be."
            )
        vo = IntVec3D(*(math.floor(s / r) for s, r in zip(bbox.start, res)))
        sz = IntVec3D(*(math.ceil(s / r) for s, r in zip(bbox.shape, res)))
        return vo, sz

    if voxel_offset is None or dataset_size is None:
        raise ValueError("Either `bbox` or (`voxel_offset` + `dataset_size`) is required")
    return IntVec3D(*voxel_offset), IntVec3D(*dataset_size)


def _validate_info_compatibility(
    existing_info: dict, new_info: dict, info_overwrite: bool
) -> None:
    """Validate that new layer params are compatible with existing info file."""
    keys_to_check = [
        "encoding",
        "resolution",
        "voxel_offset",
        "size",
        "chunk_size",
        "column_schema",
    ]
    diffs = {}
    for key in keys_to_check:
        existing_val = existing_info.get(key)
        new_val = new_info.get(key)
        if existing_val != new_val:
            diffs[key] = {"existing": existing_val, "new": new_val}

    if diffs and not info_overwrite:
        diff_str = "\n".join(f"  {k}: {v['existing']} -> {v['new']}" for k, v in diffs.items())
        raise RuntimeError(
            f"New layer parameters do not match existing info file. "
            f"Set info_overwrite=True to overwrite.\nDifferences:\n{diff_str}"
        )


@builder.register("build_volumetric_tabular_layer")
def build_volumetric_tabular_layer(
    path: str,
    resolution: Sequence[float] | None = None,
    chunk_size: Sequence[int] | None = None,
    bbox: BBox3D | None = None,
    voxel_offset: Sequence[int] | None = None,
    dataset_size: Sequence[int] | None = None,
    encoding: str = "parquet",
    column_schema: Sequence[dict[str, str]] | None = None,
    mode: Literal["read", "write", "replace"] = "write",
    delete_empty_uploads: bool = True,
    info_overwrite: bool = False,
) -> VolumetricTabularLayer:
    """Build a VolumetricTabularLayer for reading/writing chunked tabular data.

    Spatial bounds can be specified via (bbox + resolution) or
    (resolution + voxel_offset + dataset_size).

    :param path: Root directory for data files and info file.
    :param resolution: Voxel resolution in nm (x, y, z).
    :param chunk_size: Chunk size in voxels (x, y, z).
    :param bbox: Bounding box for the dataset.
    :param voxel_offset: Start of the dataset volume in voxels.
    :param dataset_size: Dataset size in voxels.
    :param encoding: File format for data files: "parquet", "csv", or "json".
    :param column_schema: List of {"name": str, "dtype": str} dicts describing columns.
        Required for write/replace mode. Specifies column names and numpy dtypes.
        Critical for CSV/JSON formats to preserve types (especially uint64).
    :param mode: How the layer should be opened:
        - "read": for reading only; info file must exist.
        - "write": for writing; writes info file, fails if it already exists.
        - "replace": for writing; clears existing data and rewrites info file.
    :param delete_empty_uploads: When True, writing an empty DataFrame deletes the chunk file.
    :param info_overwrite: When True, allow overwriting an existing info file with
        different parameters. When False, raise if new params don't match existing.
    """
    existing_info = None
    try:
        existing_info = read_info(path)
    except (FileNotFoundError, KeyError, OSError):
        pass

    if mode == "read":
        if existing_info is None:
            raise FileNotFoundError(f"Tabular layer info not found at {path}")
        backend = TabularBackend.from_path(path, delete_empty_uploads=delete_empty_uploads)
        return VolumetricTabularLayer(backend=backend, readonly=True)

    # write / replace modes
    if resolution is None:
        raise ValueError("`resolution` is required for write/replace mode")
    if chunk_size is None:
        raise ValueError("`chunk_size` is required for write/replace mode")
    if column_schema is None:
        raise ValueError("`column_schema` is required for write/replace mode")

    res = Vec3D(*resolution)
    cs = Vec3D(*chunk_size)
    vo, sz = _resolve_bounds(bbox, voxel_offset, dataset_size, res)

    schema = tuple(column_schema)

    if mode == "write" and existing_info is not None:
        new_info = {
            "encoding": encoding,
            "resolution": list(res),
            "voxel_offset": list(vo),
            "size": list(sz),
            "chunk_size": list(cs),
            "column_schema": list(schema),
        }
        _validate_info_compatibility(existing_info, new_info, info_overwrite)
        if not info_overwrite:
            # Info matches, reuse existing layer without rewriting info
            backend = TabularBackend(
                path=path,
                resolution=res,
                voxel_offset=vo,
                size=sz,
                chunk_size=cs,
                encoding=encoding,
                column_schema=schema,
                delete_empty_uploads=delete_empty_uploads,
            )
            return VolumetricTabularLayer(backend=backend)

    backend = TabularBackend(
        path=path,
        resolution=res,
        voxel_offset=vo,
        size=sz,
        chunk_size=cs,
        encoding=encoding,
        column_schema=schema,
        delete_empty_uploads=delete_empty_uploads,
    )
    backend.write_info()
    return VolumetricTabularLayer(backend=backend)
