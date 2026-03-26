import numpy as np
import pandas as pd
import pytest

from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric.index import VolumetricIndex
from zetta_utils.layer.volumetric.tabular.backend import TabularBackend, read_info
from zetta_utils.layer.volumetric.tabular.build import build_volumetric_tabular_layer

SAMPLE_SCHEMA = (
    {"name": "segment_id", "dtype": "int64"},
    {"name": "score", "dtype": "float64"},
    {"name": "label", "dtype": "object"},
)

UINT64_SCHEMA = (
    {"name": "seg_id", "dtype": "uint64"},
    {"name": "count", "dtype": "uint64"},
)


def _make_idx(start=(0, 0, 0), end=(64, 64, 40), resolution=(4, 4, 40)):
    return VolumetricIndex.from_coords(
        start_coord=start, end_coord=end, resolution=Vec3D(*resolution)
    )


def _make_sample_df():
    return pd.DataFrame(
        {
            "segment_id": [1, 2, 3, 4, 5],
            "score": [0.1, 0.2, 0.3, 0.4, 0.5],
            "label": ["a", "b", "c", "d", "e"],
        }
    )


def _make_uint64_df():
    """DataFrame with uint64 values beyond float64 precision (>2^53)."""
    return pd.DataFrame(
        {
            "seg_id": np.array(
                [
                    2 ** 53 + 1,  # 9007199254740993 - first non-exact float64 integer
                    2 ** 53 + 5,
                    2 ** 63,  # 9223372036854775808 - exceeds int64 max
                    2 ** 64 - 2,  # near uint64 max
                    0,
                ],
                dtype=np.uint64,
            ),
            "count": np.array([10, 20, 30, 40, 50], dtype=np.uint64),
        }
    )


def _make_backend(tmp_path, encoding="parquet", column_schema=(), **kwargs):
    return TabularBackend(
        path=str(tmp_path),
        resolution=Vec3D(4, 4, 40),
        voxel_offset=Vec3D(0, 0, 0),
        size=Vec3D(256, 256, 100),
        chunk_size=Vec3D(64, 64, 40),
        encoding=encoding,
        column_schema=column_schema,
        **kwargs,
    )


class TestTabularBackendParquet:
    def test_write_read(self, tmp_path):
        backend = _make_backend(tmp_path, encoding="parquet")
        idx = _make_idx()
        df = _make_sample_df()

        backend.write(idx, df)
        result = backend.read(idx)

        pd.testing.assert_frame_equal(result, df)

    def test_chunk_file_exists(self, tmp_path):
        backend = _make_backend(tmp_path, encoding="parquet")
        idx = _make_idx()
        backend.write(idx, _make_sample_df())

        chunk_file = tmp_path / "data" / "0-64_0-64_0-40.parquet"
        assert chunk_file.exists()


class TestTabularBackendCSV:
    def test_write_read(self, tmp_path):
        backend = _make_backend(tmp_path, encoding="csv", column_schema=SAMPLE_SCHEMA)
        idx = _make_idx()
        df = _make_sample_df()

        backend.write(idx, df)
        result = backend.read(idx)

        pd.testing.assert_frame_equal(result, df)

    def test_chunk_file_exists(self, tmp_path):
        backend = _make_backend(tmp_path, encoding="csv", column_schema=SAMPLE_SCHEMA)
        idx = _make_idx()
        backend.write(idx, _make_sample_df())

        chunk_file = tmp_path / "data" / "0-64_0-64_0-40.csv"
        assert chunk_file.exists()


class TestTabularBackendJSON:
    def test_write_read(self, tmp_path):
        backend = _make_backend(tmp_path, encoding="json", column_schema=SAMPLE_SCHEMA)
        idx = _make_idx()
        df = _make_sample_df()

        backend.write(idx, df)
        result = backend.read(idx)

        pd.testing.assert_frame_equal(result, df)

    def test_chunk_file_exists(self, tmp_path):
        backend = _make_backend(tmp_path, encoding="json", column_schema=SAMPLE_SCHEMA)
        idx = _make_idx()
        backend.write(idx, _make_sample_df())

        chunk_file = tmp_path / "data" / "0-64_0-64_0-40.json"
        assert chunk_file.exists()


class TestTabularBackendReadMissing:
    def test_read_missing_chunk_returns_empty(self, tmp_path):
        backend = _make_backend(tmp_path)
        idx = _make_idx()
        result = backend.read(idx)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestTabularBackendDeleteEmpty:
    def test_write_empty_deletes_chunk(self, tmp_path):
        backend = _make_backend(tmp_path, encoding="parquet")
        idx = _make_idx()

        backend.write(idx, _make_sample_df())
        chunk_file = tmp_path / "data" / "0-64_0-64_0-40.parquet"
        assert chunk_file.exists()

        backend.write(idx, pd.DataFrame())
        assert not chunk_file.exists()

    def test_write_empty_keeps_chunk_when_disabled(self, tmp_path):
        backend = _make_backend(tmp_path, encoding="parquet", delete_empty_uploads=False)
        idx = _make_idx()

        backend.write(idx, _make_sample_df())
        chunk_file = tmp_path / "data" / "0-64_0-64_0-40.parquet"
        assert chunk_file.exists()

        backend.write(idx, pd.DataFrame())
        assert chunk_file.exists()


class TestTabularBackendInfo:
    def test_info_roundtrip(self, tmp_path):
        backend = _make_backend(tmp_path, encoding="csv", column_schema=UINT64_SCHEMA)
        backend.write_info()

        info = read_info(str(tmp_path))
        assert info["type"] == "volumetric_tabular"
        assert info["encoding"] == "csv"
        assert info["resolution"] == [4, 4, 40]
        assert info["voxel_offset"] == [0, 0, 0]
        assert info["size"] == [256, 256, 100]
        assert info["chunk_size"] == [64, 64, 40]
        assert info["column_schema"] == list(UINT64_SCHEMA)

    def test_from_path(self, tmp_path):
        backend = _make_backend(tmp_path, encoding="json", column_schema=UINT64_SCHEMA)
        backend.write_info()

        loaded = TabularBackend.from_path(str(tmp_path))
        assert loaded.encoding == "json"
        assert loaded.resolution == Vec3D(4, 4, 40)
        assert loaded.voxel_offset == Vec3D(0, 0, 0)
        assert loaded.size == Vec3D(256, 256, 100)
        assert loaded.chunk_size == Vec3D(64, 64, 40)
        assert loaded.column_schema == UINT64_SCHEMA


class TestTabularBackendDelete:
    def test_delete_all(self, tmp_path):
        backend = _make_backend(tmp_path)
        backend.write_info()
        backend.write(_make_idx(), _make_sample_df())

        assert (tmp_path / "info").exists()
        assert (tmp_path / "data" / "0-64_0-64_0-40.parquet").exists()

        backend.delete()

        assert not (tmp_path / "info").exists()
        assert not (tmp_path / "data").exists()


class TestTabularBackendWithChanges:
    def test_with_changes(self, tmp_path):
        backend = _make_backend(tmp_path, encoding="parquet")
        changed = backend.with_changes(encoding="csv")
        assert changed.encoding == "csv"
        assert changed.path == backend.path


class TestTabularBackendName:
    def test_name_property(self, tmp_path):
        backend = _make_backend(tmp_path, encoding="parquet")
        assert "TabularBackend[" in backend.name
        assert str(tmp_path) in backend.name


class TestTabularBackendValidation:
    def test_invalid_encoding_raises(self, tmp_path):
        with pytest.raises(ValueError, match="encoding must be one of"):
            _make_backend(tmp_path, encoding="xml")


class TestUint64Roundtrip:
    """Verify that uint64 values > 2^53 survive serialization without precision loss."""

    @pytest.mark.parametrize("encoding", ["parquet", "csv", "json"])
    def test_uint64_roundtrip(self, tmp_path, encoding):
        backend = _make_backend(tmp_path, encoding=encoding, column_schema=UINT64_SCHEMA)
        idx = _make_idx()
        df = _make_uint64_df()

        backend.write(idx, df)
        result = backend.read(idx)

        np.testing.assert_array_equal(result["seg_id"].to_numpy(), df["seg_id"].to_numpy())
        np.testing.assert_array_equal(result["count"].to_numpy(), df["count"].to_numpy())
        assert result["seg_id"].dtype == np.uint64
        assert result["count"].dtype == np.uint64


class TestFormatConversion:
    """Write with one encoding, read back, re-write with another, verify data survives."""

    @pytest.mark.parametrize(
        "src_enc,dst_enc",
        [
            ("parquet", "csv"),
            ("parquet", "json"),
            ("csv", "parquet"),
            ("csv", "json"),
            ("json", "parquet"),
            ("json", "csv"),
        ],
    )
    def test_convert_format(self, tmp_path, src_enc, dst_enc):
        src_dir = tmp_path / "src"
        dst_dir = tmp_path / "dst"
        idx = _make_idx()
        df = _make_sample_df()

        src_backend = _make_backend(src_dir, encoding=src_enc, column_schema=SAMPLE_SCHEMA)
        src_backend.write(idx, df)
        intermediate = src_backend.read(idx)

        dst_backend = _make_backend(dst_dir, encoding=dst_enc, column_schema=SAMPLE_SCHEMA)
        dst_backend.write(idx, intermediate)
        result = dst_backend.read(idx)

        pd.testing.assert_frame_equal(result, df)

    @pytest.mark.parametrize(
        "src_enc,dst_enc",
        [
            ("parquet", "csv"),
            ("parquet", "json"),
            ("csv", "parquet"),
            ("csv", "json"),
            ("json", "parquet"),
            ("json", "csv"),
        ],
    )
    def test_convert_format_uint64(self, tmp_path, src_enc, dst_enc):
        src_dir = tmp_path / "src"
        dst_dir = tmp_path / "dst"
        idx = _make_idx()
        df = _make_uint64_df()

        src_backend = _make_backend(src_dir, encoding=src_enc, column_schema=UINT64_SCHEMA)
        src_backend.write(idx, df)
        intermediate = src_backend.read(idx)

        dst_backend = _make_backend(dst_dir, encoding=dst_enc, column_schema=UINT64_SCHEMA)
        dst_backend.write(idx, intermediate)
        result = dst_backend.read(idx)

        np.testing.assert_array_equal(result["seg_id"].to_numpy(), df["seg_id"].to_numpy())
        assert result["seg_id"].dtype == np.uint64


class TestChunkSizeConversion:
    """Write data across multiple chunks, then re-chunk into a different chunk size."""

    def test_rechunk(self, tmp_path):
        src_dir = tmp_path / "src"
        dst_dir = tmp_path / "dst"

        src_backend = TabularBackend(
            path=str(src_dir),
            resolution=Vec3D(4, 4, 40),
            voxel_offset=Vec3D(0, 0, 0),
            size=Vec3D(128, 64, 40),
            chunk_size=Vec3D(64, 64, 40),
            encoding="parquet",
        )

        df_chunk0 = pd.DataFrame({"x": [1, 2], "val": [10, 20]})
        df_chunk1 = pd.DataFrame({"x": [3, 4], "val": [30, 40]})
        idx0 = _make_idx(start=(0, 0, 0), end=(64, 64, 40))
        idx1 = _make_idx(start=(64, 0, 0), end=(128, 64, 40))

        src_backend.write(idx0, df_chunk0)
        src_backend.write(idx1, df_chunk1)

        # Re-chunk into a single larger chunk
        dst_backend = TabularBackend(
            path=str(dst_dir),
            resolution=Vec3D(4, 4, 40),
            voxel_offset=Vec3D(0, 0, 0),
            size=Vec3D(128, 64, 40),
            chunk_size=Vec3D(128, 64, 40),
            encoding="parquet",
        )

        combined = pd.concat([src_backend.read(idx0), src_backend.read(idx1)], ignore_index=True)
        dst_idx = _make_idx(start=(0, 0, 0), end=(128, 64, 40))
        dst_backend.write(dst_idx, combined)

        result = dst_backend.read(dst_idx)
        assert len(result) == 4
        assert list(result["x"]) == [1, 2, 3, 4]
        assert list(result["val"]) == [10, 20, 30, 40]

    def test_rechunk_with_format_change_uint64(self, tmp_path):
        """Rechunk AND change format simultaneously, preserving uint64."""
        src_dir = tmp_path / "src"
        dst_dir = tmp_path / "dst"

        src_backend = TabularBackend(
            path=str(src_dir),
            resolution=Vec3D(4, 4, 40),
            voxel_offset=Vec3D(0, 0, 0),
            size=Vec3D(128, 64, 40),
            chunk_size=Vec3D(128, 64, 40),
            encoding="parquet",
            column_schema=UINT64_SCHEMA,
        )

        df = _make_uint64_df()
        big_idx = _make_idx(start=(0, 0, 0), end=(128, 64, 40))
        src_backend.write(big_idx, df)

        # Split into two smaller csv chunks
        dst_backend = TabularBackend(
            path=str(dst_dir),
            resolution=Vec3D(4, 4, 40),
            voxel_offset=Vec3D(0, 0, 0),
            size=Vec3D(128, 64, 40),
            chunk_size=Vec3D(64, 64, 40),
            encoding="csv",
            column_schema=UINT64_SCHEMA,
        )

        full_data = src_backend.read(big_idx)
        half = len(full_data) // 2
        idx0 = _make_idx(start=(0, 0, 0), end=(64, 64, 40))
        idx1 = _make_idx(start=(64, 0, 0), end=(128, 64, 40))
        dst_backend.write(idx0, full_data.iloc[:half].reset_index(drop=True))
        dst_backend.write(idx1, full_data.iloc[half:].reset_index(drop=True))

        recombined = pd.concat([dst_backend.read(idx0), dst_backend.read(idx1)], ignore_index=True)
        np.testing.assert_array_equal(
            recombined["seg_id"].to_numpy(),
            df["seg_id"].to_numpy(),
        )
        assert recombined["seg_id"].dtype == np.uint64


BUILDER_SCHEMA = (
    {"name": "x", "dtype": "int64"},
    {"name": "y", "dtype": "float64"},
)


class TestTabularBackendDeleteChunkMissing:
    def test_delete_chunk_nonexistent_is_noop(self, tmp_path):
        backend = _make_backend(tmp_path, encoding="parquet")
        idx = _make_idx()
        # Should not raise even if chunk doesn't exist
        backend.delete_chunk(idx)


class TestReadInfoErrors:
    def test_read_info_missing_path(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_info(str(tmp_path / "nonexistent"))

    def test_read_info_empty_file(self, tmp_path):
        (tmp_path / "info").write_text("")
        with pytest.raises(FileNotFoundError):
            read_info(str(tmp_path))


class TestBuildVolumetricTabularLayer:
    def test_write_mode(self, tmp_path):
        layer = build_volumetric_tabular_layer(
            path=str(tmp_path),
            resolution=[4, 4, 40],
            chunk_size=[64, 64, 40],
            voxel_offset=[0, 0, 0],
            dataset_size=[256, 256, 100],
            encoding="parquet",
            column_schema=list(BUILDER_SCHEMA),
            mode="write",
        )
        assert not layer.readonly
        assert (tmp_path / "info").exists()

    def test_read_mode(self, tmp_path):
        build_volumetric_tabular_layer(
            path=str(tmp_path),
            resolution=[4, 4, 40],
            chunk_size=[64, 64, 40],
            voxel_offset=[0, 0, 0],
            dataset_size=[256, 256, 100],
            column_schema=list(BUILDER_SCHEMA),
            mode="write",
        )
        layer = build_volumetric_tabular_layer(
            path=str(tmp_path),
            mode="read",
        )
        assert layer.readonly
        assert layer.backend.column_schema == BUILDER_SCHEMA

    def test_read_mode_fails_without_info(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_volumetric_tabular_layer(
                path=str(tmp_path / "nonexistent"),
                mode="read",
            )

    def test_write_mode_matching_info_succeeds(self, tmp_path):
        """Write mode with matching existing info should succeed (idempotent)."""
        build_volumetric_tabular_layer(
            path=str(tmp_path),
            resolution=[4, 4, 40],
            chunk_size=[64, 64, 40],
            voxel_offset=[0, 0, 0],
            dataset_size=[256, 256, 100],
            column_schema=list(BUILDER_SCHEMA),
            mode="write",
        )
        # Same params again should succeed
        layer = build_volumetric_tabular_layer(
            path=str(tmp_path),
            resolution=[4, 4, 40],
            chunk_size=[64, 64, 40],
            voxel_offset=[0, 0, 0],
            dataset_size=[256, 256, 100],
            column_schema=list(BUILDER_SCHEMA),
            mode="write",
        )
        assert not layer.readonly

    def test_write_mode_mismatched_info_raises(self, tmp_path):
        """Write mode with different params should raise."""
        build_volumetric_tabular_layer(
            path=str(tmp_path),
            resolution=[4, 4, 40],
            chunk_size=[64, 64, 40],
            voxel_offset=[0, 0, 0],
            dataset_size=[256, 256, 100],
            column_schema=list(BUILDER_SCHEMA),
            mode="write",
        )
        with pytest.raises(RuntimeError, match="do not match"):
            build_volumetric_tabular_layer(
                path=str(tmp_path),
                resolution=[4, 4, 40],
                chunk_size=[64, 64, 40],
                voxel_offset=[0, 0, 0],
                dataset_size=[256, 256, 100],
                encoding="csv",  # different encoding
                column_schema=list(BUILDER_SCHEMA),
                mode="write",
            )

    def test_write_mode_mismatched_with_overwrite(self, tmp_path):
        """Write mode with different params + info_overwrite should succeed."""
        build_volumetric_tabular_layer(
            path=str(tmp_path),
            resolution=[4, 4, 40],
            chunk_size=[64, 64, 40],
            voxel_offset=[0, 0, 0],
            dataset_size=[256, 256, 100],
            column_schema=list(BUILDER_SCHEMA),
            mode="write",
        )
        layer = build_volumetric_tabular_layer(
            path=str(tmp_path),
            resolution=[4, 4, 40],
            chunk_size=[64, 64, 40],
            voxel_offset=[0, 0, 0],
            dataset_size=[256, 256, 100],
            encoding="csv",
            column_schema=list(BUILDER_SCHEMA),
            mode="write",
            info_overwrite=True,
        )
        assert layer.backend.encoding == "csv"

    def test_write_mode_requires_column_schema(self, tmp_path):
        with pytest.raises(ValueError, match="column_schema"):
            build_volumetric_tabular_layer(
                path=str(tmp_path),
                resolution=[4, 4, 40],
                chunk_size=[64, 64, 40],
                voxel_offset=[0, 0, 0],
                dataset_size=[256, 256, 100],
                mode="write",
            )

    def test_replace_mode(self, tmp_path):
        layer1 = build_volumetric_tabular_layer(
            path=str(tmp_path),
            resolution=[4, 4, 40],
            chunk_size=[64, 64, 40],
            voxel_offset=[0, 0, 0],
            dataset_size=[256, 256, 100],
            encoding="parquet",
            column_schema=list(BUILDER_SCHEMA),
            mode="write",
        )
        idx = _make_idx()
        layer1.backend.write(idx, _make_sample_df())

        layer2 = build_volumetric_tabular_layer(
            path=str(tmp_path),
            resolution=[4, 4, 40],
            chunk_size=[64, 64, 40],
            voxel_offset=[0, 0, 0],
            dataset_size=[256, 256, 100],
            encoding="csv",
            column_schema=list(SAMPLE_SCHEMA),
            mode="replace",
        )
        assert layer2.backend.encoding == "csv"
        assert not (tmp_path / "data" / "0-64_0-64_0-40.parquet").exists()

    def test_write_mode_requires_resolution(self, tmp_path):
        with pytest.raises(ValueError, match="resolution"):
            build_volumetric_tabular_layer(
                path=str(tmp_path),
                chunk_size=[64, 64, 40],
                voxel_offset=[0, 0, 0],
                dataset_size=[256, 256, 100],
                column_schema=list(BUILDER_SCHEMA),
                mode="write",
            )

    def test_write_mode_requires_chunk_size(self, tmp_path):
        with pytest.raises(ValueError, match="chunk_size"):
            build_volumetric_tabular_layer(
                path=str(tmp_path),
                resolution=[4, 4, 40],
                voxel_offset=[0, 0, 0],
                dataset_size=[256, 256, 100],
                column_schema=list(BUILDER_SCHEMA),
                mode="write",
            )

    def test_bbox_with_voxel_offset_raises(self, tmp_path):
        with pytest.raises(ValueError, match="voxel_offset"):
            build_volumetric_tabular_layer(
                path=str(tmp_path),
                resolution=[4, 4, 40],
                chunk_size=[64, 64, 40],
                bbox=BBox3D(bounds=((0, 1024), (0, 1024), (0, 4000))),
                voxel_offset=[0, 0, 0],
                column_schema=list(BUILDER_SCHEMA),
                mode="write",
            )

    def test_no_bbox_or_voxel_offset_raises(self, tmp_path):
        with pytest.raises(ValueError, match="bbox"):
            build_volumetric_tabular_layer(
                path=str(tmp_path),
                resolution=[4, 4, 40],
                chunk_size=[64, 64, 40],
                column_schema=list(BUILDER_SCHEMA),
                mode="write",
            )

    def test_build_with_bbox(self, tmp_path):
        layer = build_volumetric_tabular_layer(
            path=str(tmp_path),
            resolution=[4, 4, 40],
            chunk_size=[64, 64, 40],
            bbox=BBox3D(bounds=((0, 1024), (0, 1024), (0, 4000))),
            column_schema=list(BUILDER_SCHEMA),
            mode="write",
        )
        assert layer.backend.voxel_offset == Vec3D(0, 0, 0)
        assert layer.backend.size == Vec3D(256, 256, 100)


class TestChunkAlignmentValidation:
    def test_read_non_chunk_aligned_raises(self, tmp_path):
        backend = _make_backend(tmp_path)
        idx = _make_idx(start=(10, 0, 0), end=(74, 64, 40))
        with pytest.raises(ValueError, match="not chunk-aligned"):
            backend.read(idx)

    def test_write_non_chunk_aligned_raises(self, tmp_path):
        backend = _make_backend(tmp_path)
        idx = _make_idx(start=(10, 0, 0), end=(74, 64, 40))
        with pytest.raises(ValueError, match="not chunk-aligned"):
            backend.write(idx, _make_sample_df())

    def test_read_multi_chunk_raises(self, tmp_path):
        backend = _make_backend(tmp_path)
        idx = _make_idx(start=(0, 0, 0), end=(128, 64, 40))
        with pytest.raises(NotImplementedError, match="Multi-chunk"):
            backend.read(idx)

    def test_write_multi_chunk_raises(self, tmp_path):
        backend = _make_backend(tmp_path)
        idx = _make_idx(start=(0, 0, 0), end=(128, 64, 40))
        with pytest.raises(NotImplementedError, match="Multi-chunk"):
            backend.write(idx, _make_sample_df())

    def test_different_resolution_idx(self, tmp_path):
        """Idx at a different resolution should be converted to native and validated."""
        backend = _make_backend(tmp_path)
        idx = _make_idx()
        backend.write(idx, _make_sample_df())

        # Read with idx at 2x resolution (half the voxel coords)
        idx_2x = _make_idx(start=(0, 0, 0), end=(32, 32, 20), resolution=(8, 8, 80))
        result = backend.read(idx_2x)
        pd.testing.assert_frame_equal(result, _make_sample_df())
