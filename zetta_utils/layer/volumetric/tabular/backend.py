from __future__ import annotations

import io
import json
import os
from typing import Any, Literal

import attrs
import pandas as pd
from cloudfiles import CloudFiles

from zetta_utils.common import abspath, is_local
from zetta_utils.common.path import strip_prefix
from zetta_utils.geometry import Vec3D
from zetta_utils.layer.backend_base import Backend
from zetta_utils.layer.volumetric.index import VolumetricIndex

VALID_ENCODINGS = ("parquet", "csv", "json")


def read_info(path: str) -> dict[str, Any]:
    """Read tabular layer info file from path."""
    cf = CloudFiles(abspath(path))
    raw = cf.get("info")
    if raw is None or len(raw) == 0:
        raise FileNotFoundError(f"Info file not found at {path}/info")
    return json.loads(raw.decode("utf-8"))


def _validate_encoding(
    instance, attribute, value
):  # noqa: ARG001  # pylint: disable=unused-argument
    if value not in VALID_ENCODINGS:
        raise ValueError(f"encoding must be one of {VALID_ENCODINGS}, got {value!r}")


def _dtypes_from_column_schema(
    column_schema: tuple[dict[str, str], ...],
) -> dict[str, str]:
    return {entry["name"]: entry["dtype"] for entry in column_schema if "dtype" in entry}


@attrs.frozen
class TabularBackend(Backend[VolumetricIndex, pd.DataFrame, pd.DataFrame]):
    path: str
    resolution: Vec3D
    voxel_offset: Vec3D[int]
    size: Vec3D[int]
    chunk_size: Vec3D[int]
    encoding: str = attrs.field(default="parquet", validator=_validate_encoding)
    column_schema: tuple[dict[str, str], ...] = ()
    delete_empty_uploads: bool = True

    @property
    def name(self) -> str:
        return f"TabularBackend[{self.path}]"

    @property
    def _file_extension(self) -> str:
        return self.encoding

    def _get_cf(self) -> CloudFiles:
        return CloudFiles(abspath(self.path))

    def get_voxel_offset(self) -> Vec3D[int]:  # pragma: no cover
        return self.voxel_offset

    def get_chunk_size(self) -> Vec3D[int]:  # pragma: no cover
        return self.chunk_size

    def get_dataset_size(self) -> Vec3D[int]:  # pragma: no cover
        return self.size

    def get_bounds(self) -> VolumetricIndex:  # pragma: no cover
        return VolumetricIndex.from_coords(
            self.voxel_offset, self.voxel_offset + self.size, self.resolution
        )

    def _to_native_resolution(self, idx: VolumetricIndex) -> VolumetricIndex:
        return VolumetricIndex(bbox=idx.bbox, resolution=self.resolution)

    def get_chunk_aligned_index(
        self, idx: VolumetricIndex, mode: Literal["expand", "shrink"]
    ) -> VolumetricIndex:
        return idx.snapped(self.get_voxel_offset(), self.get_chunk_size(), mode)

    def assert_idx_is_chunk_aligned(self, idx: VolumetricIndex) -> None:
        idx_expanded_full = self.get_chunk_aligned_index(idx, mode="expand")
        idx_shrunk_full = self.get_chunk_aligned_index(idx, mode="shrink")
        bounds = self.get_bounds()
        idx_expanded_cropped = idx_expanded_full.intersection(bounds)
        idx_shrunk_cropped = idx_shrunk_full.intersection(bounds)

        if idx not in (idx_expanded_full, idx_expanded_cropped):
            raise ValueError(
                "The BBox3D of the specified VolumetricIndex is not chunk-aligned with"
                + f" the TabularBackend at `{self.name}`;\n"
                + f"in {tuple(self.resolution)} voxels:"
                + f" offset: {self.get_voxel_offset()},"
                + f" chunk_size: {self.get_chunk_size()}\n"
                + f" dataset bounds: {bounds}\n"
                + f"Received BBox3D: {idx.pformat()}\n"
                + "Nearest chunk-aligned BBox3Ds before cropping to bounds:\n"
                + f" - expanded : {idx_expanded_full.pformat()}\n"
                + f" - shrunk   : {idx_shrunk_full.pformat()}\n"
                + "Nearest chunk-aligned BBox3Ds after cropping to bounds:\n"
                + f" - expanded : {idx_expanded_cropped.pformat()}\n"
                + f" - shrunk   : {idx_shrunk_cropped.pformat()}"
            )

        shape = idx.stop - idx.start
        if shape != self.chunk_size:
            raise NotImplementedError(
                f"Multi-chunk read/write is not yet supported. "
                f"Requested shape {shape} spans more than one chunk "
                f"(chunk_size={self.chunk_size} at resolution {tuple(self.resolution)})"
            )

    def _chunk_relative_path(self, idx: VolumetricIndex) -> str:
        start = idx.start
        end = idx.stop
        fname = (
            f"{int(start[0])}-{int(end[0])}"
            f"_{int(start[1])}-{int(end[1])}"
            f"_{int(start[2])}-{int(end[2])}.{self._file_extension}"
        )
        return f"data/{fname}"

    def _serialize(self, data: pd.DataFrame) -> bytes:
        buf = io.BytesIO()
        if self.encoding == "parquet":
            data.to_parquet(buf, index=False)
        elif self.encoding == "csv":
            data.to_csv(buf, index=False)
        elif self.encoding == "json":
            data.to_json(buf, orient="records")
        buf.seek(0)
        return buf.getvalue()

    def _deserialize(self, raw: bytes) -> pd.DataFrame:
        buf = io.BytesIO(raw)
        if self.encoding == "parquet":
            return pd.read_parquet(buf)
        elif self.encoding == "csv":
            # Read as string to avoid lossy type inference for large integers
            df = pd.read_csv(buf, dtype=str)
        elif self.encoding == "json":
            df = pd.read_json(buf, orient="records", dtype=str)
        else:  # pragma: no cover
            raise ValueError(f"Unknown encoding: {self.encoding!r}")

        # Restore dtypes from column_schema (critical for uint64 in CSV/JSON)
        dtypes = _dtypes_from_column_schema(self.column_schema)
        for col, dtype_str in dtypes.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype_str)
        return df

    def write_info(self) -> None:
        """Write info file to disk."""
        info: dict[str, Any] = {
            "type": "volumetric_tabular",
            "encoding": self.encoding,
            "resolution": list(self.resolution),
            "voxel_offset": list(self.voxel_offset),
            "size": list(self.size),
            "chunk_size": list(self.chunk_size),
            "column_schema": list(self.column_schema),
        }
        if is_local(self.path):
            os.makedirs(strip_prefix(abspath(self.path)), exist_ok=True)
        cf = self._get_cf()
        cf.put(
            "info",
            json.dumps(info, indent=2).encode("utf-8"),
            cache_control="no-cache, no-store, max-age=0, must-revalidate",
        )

    @classmethod
    def from_path(cls, path: str, **overrides) -> TabularBackend:
        """Load backend from existing info file."""
        info = read_info(path)
        return cls(
            path=path,
            resolution=Vec3D(*info["resolution"]),
            voxel_offset=Vec3D(*info["voxel_offset"]),
            size=Vec3D(*info["size"]),
            chunk_size=Vec3D(*info["chunk_size"]),
            encoding=info.get("encoding", "parquet"),
            column_schema=tuple(
                {"name": e["name"], "dtype": e["dtype"]}
                for e in info.get("column_schema", [])
                if "name" in e and "dtype" in e
            ),
            **overrides,
        )

    def read(self, idx: VolumetricIndex) -> pd.DataFrame:
        native_idx = self._to_native_resolution(idx)
        self.assert_idx_is_chunk_aligned(native_idx)
        cf = self._get_cf()
        rel_path = self._chunk_relative_path(native_idx)
        raw = cf.get(rel_path)
        if raw is None or len(raw) == 0:
            return pd.DataFrame()
        return self._deserialize(raw)

    def write(self, idx: VolumetricIndex, data: pd.DataFrame) -> None:
        idx = self._to_native_resolution(idx)
        self.assert_idx_is_chunk_aligned(idx)
        if len(data) == 0 and self.delete_empty_uploads:
            self.delete_chunk(idx)
            return

        if is_local(self.path):
            data_dir = os.path.join(strip_prefix(abspath(self.path)), "data")
            os.makedirs(data_dir, exist_ok=True)

        cf = self._get_cf()
        rel_path = self._chunk_relative_path(idx)
        cf.put(
            rel_path,
            self._serialize(data),
            cache_control="no-cache, no-store, max-age=0, must-revalidate",
        )

    def delete(self) -> None:
        path = abspath(self.path)
        cf = CloudFiles(path)
        file_list = list(cf.list())
        if file_list:
            cf.delete(file_list)
        if is_local(self.path):
            local_path = strip_prefix(path)
            if os.path.isdir(local_path):
                for root, dirs, _ in os.walk(local_path, topdown=False):
                    for directory in dirs:
                        try:
                            os.rmdir(os.path.join(root, directory))
                        except OSError:  # pragma: no cover
                            pass

    def delete_chunk(self, idx: VolumetricIndex) -> None:
        cf = self._get_cf()
        rel_path = self._chunk_relative_path(idx)
        cf.delete(rel_path)

    def with_changes(self, **kwargs) -> TabularBackend:
        return attrs.evolve(self, **kwargs)

    def pformat(self) -> str:  # pragma: no cover
        return self.name
