import io
import json
import os
import struct
import tempfile

import pytest

from zetta_utils.layer.volumetric.annotation.annotations import (
    LineAnnotation,
    PointAnnotation,
    PropertySpec,
    Relationship,
    ShardingSpec,
    SpatialEntry,
)
from zetta_utils.layer.volumetric.annotation.sharding import (
    Chunk,
    get_shard_hex,
    write_shard_file,
    write_shard_files,
    write_shard_to_file,
)
from zetta_utils.layer.volumetric.annotation.simple_writer import (
    SimpleWriter,
    _line_demo,
    _point_demo,
)
from zetta_utils.layer.volumetric.annotation.utilities import (
    compressed_morton_code,
    is_local_filesystem,
    path_join,
    write_bytes,
)


# pylint: disable=too-many-public-methods,protected-access
class TestSimpleWriter:
    def test_init(self):
        """Test SimpleWriter initialization."""
        anno_type = "LINE"
        dimensions = {"x": [18, "nm"], "y": [18, "nm"], "z": [45, "nm"]}
        lower_bound = [0, 0, 0]
        upper_bound = [1000, 1000, 500]

        writer = SimpleWriter(anno_type, dimensions, lower_bound, upper_bound)

        assert writer.anno_type == anno_type
        assert writer.dimensions == dimensions
        assert writer.lower_bound == lower_bound
        assert writer.upper_bound == upper_bound
        assert not writer.annotations
        assert not writer.spatial_specs
        assert not writer.property_specs
        assert not writer.relationships
        assert writer.by_id_sharding is None

    def test_format_info_minimal(self):
        """Test format_info with minimal setup."""
        writer = SimpleWriter("POINT", {"x": [1, "nm"]}, [0, 0, 0], [100, 100, 100])

        info_str = writer.format_info()
        info_json = json.loads(info_str)

        assert info_json["@type"] == "neuroglancer_annotations_v1"
        assert info_json["annotation_type"] == "POINT"
        assert info_json["dimensions"] == {"x": [1, "nm"]}
        assert info_json["lower_bound"] == [0, 0, 0]
        assert info_json["upper_bound"] == [100, 100, 100]
        assert info_json["properties"] == []
        assert info_json["relationships"] == []
        assert info_json["spatial"] == []
        assert info_json["by_id"] == {"key": "by_id"}

    def test_format_info_with_specs(self):
        """Test format_info with properties, relationships, and spatial specs."""
        writer = SimpleWriter("LINE", {"x": [1, "nm"]}, [0, 0, 0], [100, 100, 100])

        # Add property spec
        prop_spec = PropertySpec("score", "float32", "Test score")
        writer.property_specs.append(prop_spec)

        # Add relationship
        rel = Relationship("test_relation")
        writer.relationships.append(rel)

        # Add spatial spec
        spatial_spec = SpatialEntry([50, 50, 50], [2, 2, 2], "spatial0", 10)
        writer.spatial_specs.append(spatial_spec)

        info_str = writer.format_info()
        info_json = json.loads(info_str)

        assert len(info_json["properties"]) == 1
        assert len(info_json["relationships"]) == 1
        assert len(info_json["spatial"]) == 1

    def test_format_info_with_sharding(self):
        """Test format_info with by_id sharding."""
        writer = SimpleWriter("POINT", {"x": [1, "nm"]}, [0, 0, 0], [100, 100, 100])
        sharding_spec = ShardingSpec(shard_bits=4, minishard_bits=3)
        writer.by_id_sharding = sharding_spec

        info_str = writer.format_info()
        info_json = json.loads(info_str)

        assert "sharding" in info_json["by_id"]

    def test_compile_multi_annotation_buffer_empty(self):
        """Test compiling empty annotation list."""
        writer = SimpleWriter("POINT", {"x": [1, "nm"]}, [0, 0, 0], [100, 100, 100])

        buffer_data = writer.compile_multi_annotation_buffer([])

        # Should contain just the count (0) as uint64le
        assert len(buffer_data) == 8
        count = struct.unpack("<Q", buffer_data[:8])[0]
        assert count == 0

    def test_compile_multi_annotation_buffer_with_points(self):
        """Test compiling point annotations to buffer."""
        writer = SimpleWriter("POINT", {"x": [1, "nm"]}, [0, 0, 0], [100, 100, 100])

        # Add a property spec
        prop_spec = PropertySpec("score", "float32", "Test score")
        writer.property_specs.append(prop_spec)

        # Create point annotations
        points = [
            PointAnnotation(id=1, position=(10, 20, 30), properties={"score": 0.5}),
            PointAnnotation(id=2, position=(40, 50, 60), properties={"score": 0.8}),
        ]

        buffer_data = writer.compile_multi_annotation_buffer(points)

        # Should start with count
        count = struct.unpack("<Q", buffer_data[:8])[0]
        assert count == 2

        # Should end with IDs
        id1 = struct.unpack("<Q", buffer_data[-16:-8])[0]
        id2 = struct.unpack("<Q", buffer_data[-8:])[0]
        assert id1 == 1
        assert id2 == 2

    def test_compile_multi_annotation_buffer_randomize(self):
        """Test compiling annotations with randomization."""
        writer = SimpleWriter("POINT", {"x": [1, "nm"]}, [0, 0, 0], [100, 100, 100])

        points = [PointAnnotation(id=i, position=(i, i, i)) for i in range(10)]

        # Test multiple times to check randomization works
        buffer1 = writer.compile_multi_annotation_buffer(points, randomize=True)
        buffer2 = writer.compile_multi_annotation_buffer(points, randomize=True)

        # Buffers should have same length but may have different content
        assert len(buffer1) == len(buffer2)

        # Count should be the same
        count1 = struct.unpack("<Q", buffer1[:8])[0]
        count2 = struct.unpack("<Q", buffer2[:8])[0]
        assert count1 == count2 == 10

    def test_compile_multi_annotation_buffer_uses_self_annotations(self):
        """Test that compile_multi_annotation_buffer uses self.annotations when None passed."""
        writer = SimpleWriter("POINT", {"x": [1, "nm"]}, [0, 0, 0], [100, 100, 100])

        # Add annotations to writer
        writer.annotations = [
            PointAnnotation(id=1, position=(10, 20, 30)),
            PointAnnotation(id=2, position=(40, 50, 60)),
        ]

        buffer_data = writer.compile_multi_annotation_buffer()

        count = struct.unpack("<Q", buffer_data[:8])[0]
        assert count == 2

    def test_write_annotations_to_file(self):
        """Test writing annotations to a file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_annotations")
            writer = SimpleWriter("POINT", {"x": [1, "nm"]}, [0, 0, 0], [100, 100, 100])

            points = [
                PointAnnotation(id=1, position=(10, 20, 30)),
                PointAnnotation(id=2, position=(40, 50, 60)),
            ]

            writer.write_annotations(file_path, points)

            # Verify file was created and has expected content
            assert os.path.exists(file_path)
            with open(file_path, "rb") as f:
                data = f.read()

            count = struct.unpack("<Q", data[:8])[0]
            assert count == 2

    def test_annotations_in_bounds(self):
        """Test filtering annotations by bounding box."""
        writer = SimpleWriter("POINT", {"x": [1, "nm"]}, [0, 0, 0], [100, 100, 100])

        writer.annotations = [
            PointAnnotation(id=1, position=(10, 10, 10)),  # Inside
            PointAnnotation(id=2, position=(50, 50, 50)),  # Inside
            PointAnnotation(id=3, position=(150, 150, 150)),  # Outside
        ]

        # Test with bounding box that includes first two points
        filtered = writer.annotations_in_bounds([0, 0, 0], [60, 60, 60])

        assert len(filtered) == 2
        assert all(anno.id in [1, 2] for anno in filtered)

    def test_subdivision_cell_bounds(self):
        """Test calculation of subdivision cell bounds."""
        writer = SimpleWriter("POINT", {"x": [1, "nm"]}, [0, 0, 0], [1000, 1000, 1000])

        # Add spatial spec
        spatial_spec = SpatialEntry([100, 100, 100], [10, 10, 10], "spatial0", 10)
        writer.spatial_specs.append(spatial_spec)

        start, end = writer.subdivision_cell_bounds(0, (2, 3, 1))

        expected_start = [0 + 2 * 100, 0 + 3 * 100, 0 + 1 * 100]
        expected_end = [expected_start[0] + 100, expected_start[1] + 100, expected_start[2] + 100]

        assert start == expected_start
        assert end == expected_end

    def test_write_by_id_index_unsharded(self, mocker):
        """Test writing unsharded by_id index."""
        mock_write_bytes = mocker.patch(
            "zetta_utils.layer.volumetric.annotation.simple_writer.write_bytes"
        )
        mock_path_join = mocker.patch(
            "zetta_utils.layer.volumetric.annotation.simple_writer.path_join"
        )
        mock_path_join.side_effect = lambda *args: "/".join(args)

        writer = SimpleWriter("POINT", {"x": [1, "nm"]}, [0, 0, 0], [100, 100, 100])

        # Add some annotations
        writer.annotations = [
            PointAnnotation(id=1, position=(10, 20, 30)),
            PointAnnotation(id=2, position=(40, 50, 60)),
        ]

        writer._write_by_id_index("/test/by_id")

        # Should have called write_bytes twice (once per annotation)
        assert mock_write_bytes.call_count == 2

    def test_write_by_id_index_sharded(self, mocker):
        """Test writing sharded by_id index."""
        mock_write_shard_files = mocker.patch(
            "zetta_utils.layer.volumetric.annotation.simple_writer.write_shard_files"
        )

        writer = SimpleWriter("POINT", {"x": [1, "nm"]}, [0, 0, 0], [100, 100, 100])
        writer.by_id_sharding = ShardingSpec(shard_bits=4, minishard_bits=3)

        # Add some annotations
        writer.annotations = [
            PointAnnotation(id=1, position=(10, 20, 30)),
            PointAnnotation(id=2, position=(40, 50, 60)),
        ]

        writer._write_by_id_index("/test/by_id")

        # Should have called write_shard_files once
        mock_write_shard_files.assert_called_once()

    def test_write_spatial_index(self, mocker):
        """Test writing spatial index."""
        mock_subdivide = mocker.patch.object(SimpleWriter, "subdivide")

        writer = SimpleWriter("POINT", {"x": [1, "nm"]}, [0, 0, 0], [100, 100, 100])

        # Add spatial specs
        writer.spatial_specs = [
            SpatialEntry([100, 100, 100], [1, 1, 1], "spatial0", 10),
            SpatialEntry([50, 50, 50], [2, 2, 2], "spatial1", 10),
        ]

        writer._write_spatial_index("/test/dir")

        # Should call subdivide with default probabilities
        expected_probs = [0.5, 1.0]  # (i+1)/levels for i in range(2)
        mock_subdivide.assert_called_once_with("/test/dir", expected_probs)

    def test_write_spatial_index_custom_probs(self, mocker):
        """Test writing spatial index with custom probabilities."""
        mock_subdivide = mocker.patch.object(SimpleWriter, "subdivide")

        writer = SimpleWriter("POINT", {"x": [1, "nm"]}, [0, 0, 0], [100, 100, 100])

        custom_probs = [0.3, 0.7]
        writer._write_spatial_index("/test/dir", custom_probs)

        mock_subdivide.assert_called_once_with("/test/dir", custom_probs)

    def test_write_related_index_unsharded(self, mocker):
        """Test writing related object ID index without sharding."""
        mock_write_annotations = mocker.patch.object(SimpleWriter, "write_annotations")
        mock_path_join = mocker.patch(
            "zetta_utils.layer.volumetric.annotation.simple_writer.path_join"
        )
        mock_path_join.side_effect = lambda *args: "/".join(args)

        writer = SimpleWriter("LINE", {"x": [1, "nm"]}, [0, 0, 0], [100, 100, 100])

        # Create relationship
        relation = Relationship("presyn", key="presyn_dir")

        # Add annotations with relations
        writer.annotations = [
            LineAnnotation(id=1, start=(10, 20, 30), end=(11, 21, 31), relations={"presyn": 123}),
            LineAnnotation(id=2, start=(40, 50, 60), end=(41, 51, 61), relations={"presyn": 123}),
            LineAnnotation(id=3, start=(70, 80, 90), end=(71, 81, 91), relations={"presyn": 456}),
        ]

        writer._write_related_index("/test/dir", relation)

        # Should call write_annotations twice (once per related ID)
        assert mock_write_annotations.call_count == 2

    def test_write_related_index_sharded(self, mocker):
        """Test writing related object ID index with sharding."""
        mock_write_shard_files = mocker.patch(
            "zetta_utils.layer.volumetric.annotation.simple_writer.write_shard_files"
        )
        mock_path_join = mocker.patch(
            "zetta_utils.layer.volumetric.annotation.simple_writer.path_join"
        )
        mock_path_join.side_effect = lambda *args: "/".join(args)

        writer = SimpleWriter("LINE", {"x": [1, "nm"]}, [0, 0, 0], [100, 100, 100])

        # Create relationship with sharding
        sharding_spec = ShardingSpec(shard_bits=4, minishard_bits=3)
        relation = Relationship("presyn", key="presyn_dir", sharding=sharding_spec)

        # Add annotations with relations
        writer.annotations = [
            LineAnnotation(id=1, start=(10, 20, 30), end=(11, 21, 31), relations={"presyn": 123}),
            LineAnnotation(id=2, start=(40, 50, 60), end=(41, 51, 61), relations={"presyn": 456}),
        ]

        writer._write_related_index("/test/dir", relation)

        # Should call write_shard_files once
        mock_write_shard_files.assert_called_once()

    def test_write_related_index_list_relations(self, mocker):
        """Test writing related index when relations are lists."""
        mock_write = mocker.patch.object(SimpleWriter, "write_annotations")
        mock_path_join = mocker.patch(
            "zetta_utils.layer.volumetric.annotation.simple_writer.path_join"
        )
        mock_path_join.side_effect = lambda *args: "/".join(args)

        writer = SimpleWriter("LINE", {"x": [1, "nm"]}, [0, 0, 0], [100, 100, 100])

        # Create relationship
        relation = Relationship("presyn", key="presyn_dir")

        # Add annotations with list relations
        writer.annotations = [
            LineAnnotation(
                id=1, start=(10, 20, 30), end=(11, 21, 31), relations={"presyn": [123, 456]}
            ),
            LineAnnotation(
                id=2, start=(40, 50, 60), end=(41, 51, 61), relations={"presyn": [123]}
            ),
        ]

        writer._write_related_index("/test/dir", relation)

        # Should call write_annotations twice (once each for 123 and 456)
        assert mock_write.call_count == 2

    def test_write_full(self, mocker):
        """Test the main write method."""
        mock_write_bytes = mocker.patch(
            "zetta_utils.layer.volumetric.annotation.simple_writer.write_bytes"
        )
        mock_write_by_id = mocker.patch.object(SimpleWriter, "_write_by_id_index")
        mock_write_spatial = mocker.patch.object(SimpleWriter, "_write_spatial_index")
        mock_write_related = mocker.patch.object(SimpleWriter, "_write_related_index")
        mock_path_join = mocker.patch(
            "zetta_utils.layer.volumetric.annotation.simple_writer.path_join"
        )
        mock_path_join.side_effect = lambda *args: "/".join(args)

        writer = SimpleWriter("LINE", {"x": [1, "nm"]}, [0, 0, 0], [100, 100, 100])

        # Add a relationship so _write_related_index gets called
        relation = Relationship("presyn", key="presyn_dir")
        writer.relationships.append(relation)

        writer.write("/test/dir")

        # Verify all methods were called
        mock_write_by_id.assert_called_once_with("/test/dir/by_id")
        mock_write_spatial.assert_called_once_with("/test/dir")
        mock_write_related.assert_called_once_with("/test/dir", relation)
        mock_write_bytes.assert_called_once_with(
            "/test/dir/info", writer.format_info().encode("utf-8")
        )

    def test_subdivide_validation_errors(self):
        """Test subdivide method validation errors."""
        writer = SimpleWriter("POINT", {"x": [1, "nm"]}, [0, 0, 0], [100, 100, 100])

        # Test error when level 0 grid_shape is not (1,1,1)
        spatial_spec = SpatialEntry([50, 50, 50], [2, 2, 2], "spatial0", 10)
        writer.spatial_specs.append(spatial_spec)

        with pytest.raises(
            ValueError, match="subdivide requires level 0 grid_shape to be \\(1,1,1\\)"
        ):
            writer.subdivide("/test", [1.0])

    def test_subdivide_prob_length_mismatch(self):
        """Test subdivide with probability length mismatch."""
        writer = SimpleWriter("POINT", {"x": [1, "nm"]}, [0, 0, 0], [100, 100, 100])

        spatial_spec = SpatialEntry([100, 100, 100], [1, 1, 1], "spatial0", 10)
        writer.spatial_specs.append(spatial_spec)

        # Wrong number of probabilities
        with pytest.raises(ValueError, match="prob_per_level needs 1"):
            writer.subdivide("/test", [0.5, 0.8])  # Too many probs

    def test_subdivide_basic(self, mocker):
        """Test basic subdivide functionality."""
        _mock_validate = mocker.patch(
            "zetta_utils.layer.volumetric.annotation.annotations.validate_spatial_entries"
        )
        mock_write_annotations = mocker.patch.object(SimpleWriter, "write_annotations")
        _mock_write_shard_files = mocker.patch(
            "zetta_utils.layer.volumetric.annotation.simple_writer.write_shard_files"
        )
        mock_random = mocker.patch("zetta_utils.layer.volumetric.annotation.simple_writer.random")
        mock_random.return_value = 0.1

        writer = SimpleWriter("POINT", {"x": [1, "nm"]}, [0, 0, 0], [1000, 1000, 1000])

        # Add spatial spec with proper grid shape
        spatial_spec = SpatialEntry([1000, 1000, 1000], [1, 1, 1], "spatial0", 10)
        writer.spatial_specs.append(spatial_spec)

        # Add some annotations
        writer.annotations = [
            PointAnnotation(id=1, position=(100, 100, 100)),
            PointAnnotation(id=2, position=(200, 200, 200)),
        ]

        writer.subdivide("/test", [1.0])

        # Should call write_annotations once for the spatial0 level
        mock_write_annotations.assert_called_once()

    def test_subdivide_no_dir_path(self, mocker):
        """Test subdivide with None dir_path."""
        _mock_validate = mocker.patch(
            "zetta_utils.layer.volumetric.annotation.annotations.validate_spatial_entries"
        )

        writer = SimpleWriter("POINT", {"x": [1, "nm"]}, [0, 0, 0], [1000, 1000, 1000])

        # Add spatial spec
        spatial_spec = SpatialEntry([1000, 1000, 1000], [1, 1, 1], "spatial0", 10)
        writer.spatial_specs.append(spatial_spec)

        # Add some annotations
        writer.annotations = [
            PointAnnotation(id=1, position=(100, 100, 100)),
        ]

        # Should not raise error with None dir_path
        writer.subdivide(None, [1.0])

    def test_demo_functions(self):
        """Test the demo functions don't crash."""
        # Test that they can be imported and are callable
        assert callable(_line_demo)
        assert callable(_point_demo)

    def test_subdivide_with_sharding(self, mocker):
        """Test subdivide functionality with sharded spatial specs."""
        _mock_validate = mocker.patch(
            "zetta_utils.layer.volumetric.annotation.annotations.validate_spatial_entries"
        )
        mock_write_shard_files = mocker.patch(
            "zetta_utils.layer.volumetric.annotation.simple_writer.write_shard_files"
        )
        mock_compressed_morton_code = mocker.patch(
            "zetta_utils.layer.volumetric.annotation.simple_writer.compressed_morton_code"
        )
        mock_compressed_morton_code.return_value = 123
        mock_random = mocker.patch("zetta_utils.layer.volumetric.annotation.simple_writer.random")
        mock_random.return_value = 0.1  # Always emit

        writer = SimpleWriter("POINT", {"x": [1, "nm"]}, [0, 0, 0], [1000, 1000, 1000])

        # Add spatial spec with sharding
        sharding_spec = ShardingSpec(shard_bits=4, minishard_bits=3)
        spatial_spec = SpatialEntry(
            [1000, 1000, 1000], [1, 1, 1], "spatial0", 10, sharding=sharding_spec
        )
        writer.spatial_specs.append(spatial_spec)

        # Add some annotations
        writer.annotations = [
            PointAnnotation(id=1, position=(100, 100, 100)),
            PointAnnotation(id=2, position=(200, 200, 200)),
        ]

        writer.subdivide("/test", [1.0])

        # Should call write_shard_files for the sharded spatial spec
        mock_write_shard_files.assert_called_once()

    def test_compile_multi_annotation_buffer_with_lines(self):
        """Test compiling line annotations to buffer."""
        writer = SimpleWriter("LINE", {"x": [1, "nm"]}, [0, 0, 0], [100, 100, 100])

        # Create line annotations
        lines = [
            LineAnnotation(id=1, start=(10, 20, 30), end=(15, 25, 35)),
            LineAnnotation(id=2, start=(40, 50, 60), end=(45, 55, 65)),
        ]

        buffer_data = writer.compile_multi_annotation_buffer(lines)

        # Should start with count
        count = struct.unpack("<Q", buffer_data[:8])[0]
        assert count == 2

        # Should be non-trivial length (more than just count and IDs)
        assert len(buffer_data) > 8 + 16  # count + 2 IDs

    def test_write_annotations_with_randomization(self):
        """Test write_annotations with randomize flag."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_annotations_random")
            writer = SimpleWriter("POINT", {"x": [1, "nm"]}, [0, 0, 0], [100, 100, 100])

            points = [PointAnnotation(id=i, position=(i * 10, i * 10, i * 10)) for i in range(5)]

            # Test with randomization enabled
            writer.write_annotations(file_path, points, randomize=True)

            # Verify file was created
            assert os.path.exists(file_path)
            with open(file_path, "rb") as f:
                data = f.read()

            count = struct.unpack("<Q", data[:8])[0]
            assert count == 5

    def test_subdivide_probability_skip_line_318(self, mocker):
        """Test subdivide skips annotations when random() > p (line 318)."""
        _mock_validate = mocker.patch(
            "zetta_utils.layer.volumetric.annotation.annotations.validate_spatial_entries"
        )
        mock_write_annotations = mocker.patch.object(SimpleWriter, "write_annotations")
        mock_random = mocker.patch("zetta_utils.layer.volumetric.annotation.simple_writer.random")
        mock_random.return_value = 0.9  # > any reasonable probability to hit continue

        writer = SimpleWriter("POINT", {"x": [1, "nm"]}, [0, 0, 0], [1000, 1000, 1000])

        # Add spatial spec
        spatial_spec = SpatialEntry([1000, 1000, 1000], [1, 1, 1], "spatial0", 10)
        writer.spatial_specs.append(spatial_spec)

        # Add annotations
        writer.annotations = [
            PointAnnotation(id=1, position=(100, 100, 100)),
            PointAnnotation(id=2, position=(200, 200, 200)),
        ]

        writer.subdivide("/test", [0.5])  # Low probability

        # Should still call write_annotations, but with fewer annotations due to skipping
        mock_write_annotations.assert_called_once()

    def test_subdivide_multi_level_child_subdivision(self, mocker):
        """Test subdivide with multiple levels to cover child subdivision logic (lines 339-383)."""
        _mock_validate = mocker.patch(
            "zetta_utils.layer.volumetric.annotation.annotations.validate_spatial_entries"
        )
        mock_write_annotations = mocker.patch.object(SimpleWriter, "write_annotations")
        mock_get_child_cell_ranges = mocker.patch(
            "zetta_utils.layer.volumetric.annotation.annotations.get_child_cell_ranges"
        )
        mock_get_child_cell_ranges.return_value = [(0, 2), (0, 2), (0, 1)]  # x, y, z ranges

        # Mock random to return different values to ensure some annotations stay for subdivision
        mock_random = mocker.patch("zetta_utils.layer.volumetric.annotation.simple_writer.random")
        # First few calls return high values (don't emit), later calls return low values (do emit)
        mock_random.side_effect = [0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1]

        writer = SimpleWriter("POINT", {"x": [1, "nm"]}, [0, 0, 0], [1000, 1000, 1000])

        # Add two spatial specs for multi-level subdivision
        spatial_spec0 = SpatialEntry([1000, 1000, 1000], [1, 1, 1], "spatial0", 10)
        spatial_spec1 = SpatialEntry([500, 500, 500], [2, 2, 2], "spatial1", 10)
        writer.spatial_specs.extend([spatial_spec0, spatial_spec1])

        # Add annotations in different locations to trigger child cell creation
        writer.annotations = [
            PointAnnotation(id=1, position=(100, 100, 100)),
            PointAnnotation(id=2, position=(600, 600, 400)),
            PointAnnotation(id=3, position=(900, 900, 900)),
        ]

        writer.subdivide("/test", [0.5, 1.0])  # Some prob for level 0, full prob for level 1

        # With the mock setup, some annotations should be left for level 1 processing
        # Should call write_annotations at least once (could be more if child cells created)
        assert mock_write_annotations.call_count >= 1

    def test_line_demo_function(self, mocker):
        """Test the _line_demo function (lines 398-457)."""
        # Mock the write method to avoid actual file I/O
        mock_write = mocker.patch.object(SimpleWriter, "write")
        mock_print = mocker.patch("builtins.print")

        # Call the demo function
        test_path = "/test/demo/path"
        _line_demo(test_path)

        # Verify write was called with the test path
        mock_write.assert_called_once_with(test_path)
        mock_print.assert_called_once_with(f"Wrote {test_path}")

    def test_point_demo_function(self, mocker):
        """Test the _point_demo function (lines 462-513)."""
        # Mock the write method and expanduser
        mock_write = mocker.patch.object(SimpleWriter, "write")
        mock_expanduser = mocker.patch("os.path.expanduser")
        mock_expanduser.return_value = "/home/user/temp/simple_anno_points"
        mock_print = mocker.patch("builtins.print")

        # Call the demo function (note: path parameter is ignored by the function)
        _point_demo("unused_parameter")

        # Verify expanduser and write were called with the hardcoded path
        mock_expanduser.assert_called_once_with("~/temp/simple_anno_points")
        mock_write.assert_called_once_with("/home/user/temp/simple_anno_points")
        mock_print.assert_called_once_with("Wrote /home/user/temp/simple_anno_points")

    # Test utilities.py functions
    def test_is_local_filesystem(self):
        """Test is_local_filesystem function."""
        assert is_local_filesystem("/local/path")
        assert is_local_filesystem("relative/path")
        assert is_local_filesystem("file:///local/path")
        assert not is_local_filesystem("gs://bucket/path")
        assert not is_local_filesystem("s3://bucket/path")

    def test_path_join_local(self):
        """Test path_join with local filesystem paths."""
        result = path_join("/base", "sub", "file.txt")
        expected = os.path.join("/base", "sub", "file.txt")
        assert result == expected

    def test_path_join_cloud(self):
        """Test path_join with cloud storage paths."""
        result = path_join("gs://bucket", "folder", "file.txt")
        assert result == "gs://bucket/folder/file.txt"

    def test_path_join_empty_raises(self):
        """Test path_join raises ValueError with no arguments."""
        with pytest.raises(ValueError, match="At least one path is required"):
            path_join()

    def test_compressed_morton_code(self):
        """Test compressed_morton_code function."""
        # Test basic case
        result = compressed_morton_code((1, 2, 3), (4, 4, 4))
        assert isinstance(result, int)

        # Test edge case with different grid shapes
        result = compressed_morton_code((0, 0, 0), (2, 3, 4))
        assert result == 0

        # Test another case
        result = compressed_morton_code((1, 1, 1), (2, 2, 2))
        assert isinstance(result, int)

    def test_write_bytes_local(self, mocker):
        """Test write_bytes with local file path."""
        mock_cloudfile = mocker.patch(
            "zetta_utils.layer.volumetric.annotation.utilities.CloudFile"
        )
        mock_cf_instance = mocker.MagicMock()
        mock_cloudfile.return_value = mock_cf_instance

        test_data = b"test data"
        write_bytes("/local/path/file.txt", test_data)

        mock_cloudfile.assert_called_once_with("file:///local/path/file.txt")
        mock_cf_instance.put.assert_called_once_with(
            test_data, cache_control="no-cache, no-store, max-age=0, must-revalidate"
        )

    def test_write_bytes_cloud(self, mocker):
        """Test write_bytes with cloud storage path."""
        mock_cloudfile = mocker.patch(
            "zetta_utils.layer.volumetric.annotation.utilities.CloudFile"
        )
        mock_cf_instance = mocker.MagicMock()
        mock_cloudfile.return_value = mock_cf_instance

        test_data = b"test data"
        write_bytes("gs://bucket/path/file.txt", test_data)

        mock_cloudfile.assert_called_once_with("gs://bucket/path/file.txt")
        mock_cf_instance.put.assert_called_once_with(
            test_data, cache_control="no-cache, no-store, max-age=0, must-revalidate"
        )

    # Test sharding.py functions
    def test_get_shard_hex(self):
        """Test get_shard_hex function."""
        # Test various shard numbers and bits
        assert get_shard_hex(0, 4) == "0"
        assert get_shard_hex(15, 4) == "f"
        assert get_shard_hex(255, 8) == "ff"
        assert get_shard_hex(10, 8) == "0a"

    def test_chunk_dataclass(self):
        """Test Chunk dataclass."""
        chunk = Chunk(chunk_id=123, data=b"test data")
        assert chunk.chunk_id == 123
        assert chunk.data == b"test data"

    def test_write_shard_file_basic(self):
        """Test write_shard_file with basic chunks."""
        output = io.BytesIO()
        sharding_spec = ShardingSpec(shard_bits=4, minishard_bits=2)
        chunks = [
            Chunk(chunk_id=0, data=b"chunk0"),
            Chunk(chunk_id=1, data=b"chunk1"),
        ]

        write_shard_file(output, sharding_spec, 0, chunks)

        # Verify output was written
        result = output.getvalue()
        assert len(result) > 0

        # Should start with shard index (minishard count * 16 bytes)
        num_minishards = sharding_spec.num_minishards_per_shard
        expected_shard_index_size = num_minishards * 16
        assert len(result) >= expected_shard_index_size

    def test_write_shard_file_multiple_chunks_per_minishard(self):
        """Test write_shard_file with multiple chunks in the same minishard to cover line 123."""
        output = io.BytesIO()
        sharding_spec = ShardingSpec(shard_bits=4, minishard_bits=2)

        # Create chunks that map to the same minishard (same last 2 bits) but same shard
        # chunk_id = 1 (minishard 1) and chunk_id = 65 (minishard 1), both in shard 0
        chunks = [
            Chunk(chunk_id=1, data=b"chunk1"),  # shard 0, minishard 1
            Chunk(chunk_id=65, data=b"chunk65"),  # shard 0, minishard 1
        ]

        write_shard_file(output, sharding_spec, 0, chunks)

        # Verify output was written
        result = output.getvalue()
        assert len(result) > 0

        # Should start with shard index (minishard count * 16 bytes)
        num_minishards = sharding_spec.num_minishards_per_shard
        expected_shard_index_size = num_minishards * 16
        assert len(result) >= expected_shard_index_size

    def test_write_shard_file_wrong_shard(self):
        """Test write_shard_file raises error for chunk in wrong shard."""
        output = io.BytesIO()
        sharding_spec = ShardingSpec(shard_bits=4, minishard_bits=2)

        # Create a chunk that belongs to shard 1, but try to write to shard 0
        chunk_id = 16  # This should be in shard 1 with 4 shard bits
        chunks = [Chunk(chunk_id=chunk_id, data=b"test")]

        with pytest.raises(ValueError, match="belongs to shard"):
            write_shard_file(output, sharding_spec, 0, chunks)

    def test_write_shard_file_empty_chunks(self):
        """Test write_shard_file with empty chunks list."""
        output = io.BytesIO()
        sharding_spec = ShardingSpec(shard_bits=4, minishard_bits=2)
        chunks: list[Chunk] = []

        write_shard_file(output, sharding_spec, 0, chunks)

        result = output.getvalue()
        # Should still write shard index even with no chunks
        num_minishards = sharding_spec.num_minishards_per_shard
        expected_size = num_minishards * 16
        assert len(result) == expected_size

    def test_write_shard_file_gzip_encoding(self):
        """Test write_shard_file with gzip data encoding."""
        output = io.BytesIO()
        sharding_spec = ShardingSpec(shard_bits=4, minishard_bits=2, data_encoding="gzip")
        chunks = [Chunk(chunk_id=0, data=b"test data to compress")]

        write_shard_file(output, sharding_spec, 0, chunks)

        result = output.getvalue()
        assert len(result) > 0

    def test_write_shard_file_gzip_minishard_index(self):
        """Test write_shard_file with gzip minishard index encoding."""
        output = io.BytesIO()
        sharding_spec = ShardingSpec(
            shard_bits=4, minishard_bits=2, minishard_index_encoding="gzip"
        )
        chunks = [Chunk(chunk_id=0, data=b"test")]

        write_shard_file(output, sharding_spec, 0, chunks)

        result = output.getvalue()
        assert len(result) > 0

    def test_write_shard_to_file(self, mocker):
        """Test write_shard_to_file function."""
        mock_open = mocker.patch("builtins.open", mocker.mock_open())
        mock_expanduser = mocker.patch("os.path.expanduser")
        mock_expanduser.return_value = "/expanded/path/test.shard"

        sharding_spec = ShardingSpec(shard_bits=4, minishard_bits=2)
        chunks = [Chunk(chunk_id=0, data=b"test")]

        write_shard_to_file("/test/path/test.shard", sharding_spec, 0, chunks)

        mock_expanduser.assert_called_once_with("/test/path/test.shard")
        mock_open.assert_called_once_with("/expanded/path/test.shard", "wb")

    def test_write_shard_files(self, mocker):
        """Test write_shard_files function."""
        mock_makedirs = mocker.patch("os.makedirs")
        mock_expanduser = mocker.patch("os.path.expanduser")
        mock_expanduser.return_value = "/expanded/dir"
        mock_write_shard_to_file = mocker.patch(
            "zetta_utils.layer.volumetric.annotation.sharding.write_shard_to_file"
        )

        sharding_spec = ShardingSpec(shard_bits=2, minishard_bits=1)  # 4 shards total
        chunks = [
            Chunk(chunk_id=0, data=b"chunk0"),  # shard 0
            Chunk(chunk_id=1, data=b"chunk1"),  # shard 1
            Chunk(chunk_id=4, data=b"chunk4"),  # shard 0
        ]

        write_shard_files("/test/dir", sharding_spec, chunks)

        mock_expanduser.assert_called_once_with("/test/dir")
        mock_makedirs.assert_called_once_with("/expanded/dir", exist_ok=True)

        # Should call write_shard_to_file for each shard (4 total)
        assert mock_write_shard_to_file.call_count == 4

    def test_write_shard_files_chunk_distribution(self, mocker):
        """Test write_shard_files correctly distributes chunks to shards."""
        mocker.patch("os.makedirs")
        mock_expanduser = mocker.patch("os.path.expanduser")
        mock_expanduser.return_value = "/expanded/dir"
        mock_write_shard_to_file = mocker.patch(
            "zetta_utils.layer.volumetric.annotation.sharding.write_shard_to_file"
        )

        sharding_spec = ShardingSpec(shard_bits=1, minishard_bits=1)  # 2 shards total
        chunks = [
            Chunk(chunk_id=0, data=b"chunk0"),  # should go to shard 0
            Chunk(chunk_id=2, data=b"chunk2"),  # should go to shard 0
            Chunk(chunk_id=1, data=b"chunk1"),  # should go to shard 1
        ]

        write_shard_files("/test/dir", sharding_spec, chunks)

        # Verify calls for both shards
        calls = mock_write_shard_to_file.call_args_list
        assert len(calls) == 2

        # Check shard 0 call
        shard_0_call = calls[0]
        assert shard_0_call[0][1] == sharding_spec  # sharding_spec
        assert shard_0_call[0][2] == 0  # shard_number
        shard_0_chunks = shard_0_call[0][3]  # chunks
        assert len(shard_0_chunks) == 2  # chunk0 and chunk2

        # Check shard 1 call
        shard_1_call = calls[1]
        assert shard_1_call[0][2] == 1  # shard_number
        shard_1_chunks = shard_1_call[0][3]  # chunks
        assert len(shard_1_chunks) == 1  # chunk1

    def test_sharded_format_integration(self, mocker):
        """Integration test showing sharded format writing in SimpleWriter."""
        mock_write_shard_files = mocker.patch(
            "zetta_utils.layer.volumetric.annotation.simple_writer.write_shard_files"
        )

        writer = SimpleWriter("POINT", {"x": [1, "nm"]}, [0, 0, 0], [100, 100, 100])
        writer.by_id_sharding = ShardingSpec(shard_bits=4, minishard_bits=3)

        # Add annotations
        writer.annotations = [
            PointAnnotation(id=1, position=(10, 20, 30)),
            PointAnnotation(id=2, position=(40, 50, 60)),
            PointAnnotation(id=15, position=(70, 80, 90)),
        ]

        writer._write_by_id_index("/test/by_id")

        # Should have called write_shard_files once
        mock_write_shard_files.assert_called_once()

        # Check the call arguments
        call_args = mock_write_shard_files.call_args
        dir_path = call_args[0][0]
        sharding_spec = call_args[0][1]
        chunks = call_args[0][2]

        assert dir_path == "/test/by_id"
        assert isinstance(sharding_spec, ShardingSpec)
        assert len(chunks) == 3  # One chunk per annotation

        # Check chunk structure
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.chunk_id in [1, 2, 15]
            assert isinstance(chunk.data, bytes)

    def test_spatial_sharded_format_integration(self, mocker):
        """Integration test for sharded spatial index writing."""
        _mock_validate = mocker.patch(
            "zetta_utils.layer.volumetric.annotation.annotations.validate_spatial_entries"
        )
        mock_write_shard_files = mocker.patch(
            "zetta_utils.layer.volumetric.annotation.simple_writer.write_shard_files"
        )
        mock_compressed_morton_code = mocker.patch(
            "zetta_utils.layer.volumetric.annotation.simple_writer.compressed_morton_code"
        )
        mock_compressed_morton_code.return_value = 123
        mock_random = mocker.patch("zetta_utils.layer.volumetric.annotation.simple_writer.random")
        mock_random.return_value = 0.1  # Always emit

        writer = SimpleWriter("POINT", {"x": [1, "nm"]}, [0, 0, 0], [1000, 1000, 1000])

        # Add spatial spec with sharding
        sharding_spec = ShardingSpec(shard_bits=4, minishard_bits=3)
        spatial_spec = SpatialEntry(
            [1000, 1000, 1000], [1, 1, 1], "spatial0", 10, sharding=sharding_spec
        )
        writer.spatial_specs.append(spatial_spec)

        # Add annotations
        writer.annotations = [
            PointAnnotation(id=1, position=(100, 100, 100)),
            PointAnnotation(id=2, position=(200, 200, 200)),
        ]

        writer.subdivide("/test", [1.0])

        # Should call write_shard_files for the sharded spatial spec
        mock_write_shard_files.assert_called_once()

        # Check the call arguments
        call_args = mock_write_shard_files.call_args
        chunks = call_args[0][2]

        # Should have at least one chunk for the spatial data
        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert isinstance(chunk.data, bytes)
