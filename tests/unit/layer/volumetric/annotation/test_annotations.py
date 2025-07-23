"""Tests for layer.volumetric.annotation.annotations module."""
# pylint: disable=too-many-lines,protected-access,use-implicit-booleaness-not-comparison

import io
import json
import struct

import pytest

from zetta_utils.geometry import Vec3D
from zetta_utils.geometry.bbox import BBox3D
from zetta_utils.layer.volumetric.annotation.annotations import (
    Annotation,
    LineAnnotation,
    PointAnnotation,
    PropertySpec,
    Relationship,
    ShardingSpec,
    SpatialEntry,
    get_child_cell_ranges,
    validate_spatial_entries,
)


class TestShardingSpec:
    """Test ShardingSpec class."""

    def test_default_initialization(self):
        """Test default initialization."""
        spec = ShardingSpec()
        assert spec.preshift_bits == 0
        assert spec.hash == "identity"
        assert spec.minishard_bits == 3
        assert spec.shard_bits == 4
        assert spec.minishard_index_encoding == "raw"
        assert spec.data_encoding == "raw"

    def test_type_property(self):
        """Test type property always returns the correct value."""
        spec = ShardingSpec()
        assert spec.type == "neuroglancer_uint64_sharded_v1"

    def test_num_shards_property(self):
        """Test num_shards property calculation."""
        spec = ShardingSpec(shard_bits=4)
        assert spec.num_shards == 16  # 2**4

        spec = ShardingSpec(shard_bits=3)
        assert spec.num_shards == 8  # 2**3

    def test_num_minishards_per_shard_property(self):
        """Test num_minishards_per_shard property calculation."""
        spec = ShardingSpec(minishard_bits=3)
        assert spec.num_minishards_per_shard == 8  # 2**3

        spec = ShardingSpec(minishard_bits=2)
        assert spec.num_minishards_per_shard == 4  # 2**2

    def test_get_shard_number_identity_hash(self):
        """Test shard number calculation with identity hash."""
        spec = ShardingSpec(preshift_bits=0, minishard_bits=3, shard_bits=4)

        # Test with chunk_id = 0b11111111 (255)
        # hashed_chunk_id = 255 >> 0 = 255
        # shard_number = (255 >> 3) & ((1 << 4) - 1) = 31 & 15 = 15
        assert spec.get_shard_number(255) == 15

        # Test with preshift_bits
        spec = ShardingSpec(preshift_bits=2, minishard_bits=3, shard_bits=4)
        # hashed_chunk_id = 255 >> 2 = 63
        # shard_number = (63 >> 3) & 15 = 7 & 15 = 7
        assert spec.get_shard_number(255) == 7

    def test_get_minishard_number_identity_hash(self):
        """Test minishard number calculation with identity hash."""
        spec = ShardingSpec(preshift_bits=0, minishard_bits=3, shard_bits=4)

        # Test with chunk_id = 0b11111111 (255)
        # hashed_chunk_id = 255 >> 0 = 255
        # minishard_number = 255 & ((1 << 3) - 1) = 255 & 7 = 7
        assert spec.get_minishard_number(255) == 7

        # Test with preshift_bits
        spec = ShardingSpec(preshift_bits=2, minishard_bits=3, shard_bits=4)
        # hashed_chunk_id = 255 >> 2 = 63
        # minishard_number = 63 & 7 = 7
        assert spec.get_minishard_number(255) == 7

    def test_get_shard_filename(self):
        """Test shard filename generation."""
        spec = ShardingSpec(shard_bits=4)  # 16 shards, width = (4+3)//4 = 1
        assert spec.get_shard_filename(0) == "0.shard"
        assert spec.get_shard_filename(15) == "f.shard"

        spec = ShardingSpec(shard_bits=8)  # 256 shards, width = (8+3)//4 = 2
        assert spec.get_shard_filename(0) == "00.shard"
        assert spec.get_shard_filename(255) == "ff.shard"

    def test_get_shard_filename_out_of_range(self):
        """Test shard filename with out of range shard number."""
        spec = ShardingSpec(shard_bits=4)  # 16 shards (0-15)
        with pytest.raises(ValueError, match="Shard number 16 out of range"):
            spec.get_shard_filename(16)

        with pytest.raises(ValueError, match="Shard number -1 out of range"):
            spec.get_shard_filename(-1)

    def test_get_shard_filename_width_calculation(self):
        """Test shard filename width calculation for different shard_bits."""
        spec = ShardingSpec(shard_bits=1)  # width = (1+3)//4 = 1
        assert spec.get_shard_filename(0) == "0.shard"

        spec = ShardingSpec(shard_bits=5)  # width = (5+3)//4 = 2
        assert spec.get_shard_filename(0) == "00.shard"

    def test_to_dict_default(self):
        """Test conversion to dictionary with default values."""
        spec = ShardingSpec()
        result = spec.to_dict()
        expected = {
            "@type": "neuroglancer_uint64_sharded_v1",
            "preshift_bits": 0,
            "hash": "identity",
            "minishard_bits": 3,
            "shard_bits": 4,
        }
        assert result == expected

    def test_to_dict_with_non_default_encodings(self):
        """Test conversion to dictionary with non-default encodings."""
        spec = ShardingSpec(minishard_index_encoding="gzip", data_encoding="gzip")
        result = spec.to_dict()
        assert result["minishard_index_encoding"] == "gzip"
        assert result["data_encoding"] == "gzip"

    def test_to_json(self):
        """Test JSON serialization."""
        spec = ShardingSpec()
        json_str = spec.to_json()
        parsed = json.loads(json_str)
        assert parsed["@type"] == "neuroglancer_uint64_sharded_v1"
        assert parsed["preshift_bits"] == 0

        # Test with indentation
        json_str = spec.to_json(indent=2)
        assert "\n" in json_str


class TestPropertySpec:
    """Test PropertySpec class."""

    def test_valid_initialization(self):
        """Test valid property spec initialization."""
        spec = PropertySpec(id="test_prop", type="uint32")
        assert spec.id == "test_prop"
        assert spec.type == "uint32"
        assert spec.description is None
        assert spec.enum_values is None
        assert spec.enum_labels is None

    def test_invalid_id_validation(self):
        """Test invalid ID validation."""
        # ID must start with lowercase letter
        with pytest.raises(ValueError, match="must match pattern"):
            PropertySpec(id="Test_prop", type="uint32")

        # ID cannot start with number
        with pytest.raises(ValueError, match="must match pattern"):
            PropertySpec(id="1test", type="uint32")

        # ID cannot contain special characters
        with pytest.raises(ValueError, match="must match pattern"):
            PropertySpec(id="test-prop", type="uint32")

    def test_invalid_type_validation(self):
        """Test invalid type validation."""
        with pytest.raises(ValueError, match="must be one of"):
            PropertySpec(id="test", type="invalid_type")

    def test_enum_validation_both_or_neither(self):
        """Test that enum_values and enum_labels must both be provided or both be None."""
        # Only enum_values provided
        with pytest.raises(ValueError, match="must both be provided or both be None"):
            PropertySpec(id="test", type="uint32", enum_values=[1, 2, 3])

        # Only enum_labels provided
        with pytest.raises(ValueError, match="must both be provided or both be None"):
            PropertySpec(id="test", type="uint32", enum_labels=["a", "b", "c"])

    def test_enum_validation_numeric_types_only(self):
        """Test that enums are only allowed for numeric types."""
        with pytest.raises(ValueError, match="enum_values not supported for type 'rgb'"):
            PropertySpec(id="test", type="rgb", enum_values=[1, 2, 3], enum_labels=["a", "b", "c"])

    def test_enum_validation_length_mismatch(self):
        """Test that enum_values and enum_labels must have same length."""
        with pytest.raises(ValueError, match="must have same length"):
            PropertySpec(id="test", type="uint32", enum_values=[1, 2, 3], enum_labels=["a", "b"])

    def test_to_dict_minimal(self):
        """Test conversion to dictionary with minimal properties."""
        spec = PropertySpec(id="test", type="uint32")
        result = spec.to_dict()
        expected = {"id": "test", "type": "uint32"}
        assert result == expected

    def test_to_dict_with_description(self):
        """Test conversion to dictionary with description."""
        spec = PropertySpec(id="test", type="uint32", description="Test property")
        result = spec.to_dict()
        assert result["description"] == "Test property"

    def test_to_dict_with_enums(self):
        """Test conversion to dictionary with enums."""
        spec = PropertySpec(
            id="test", type="uint32", enum_values=[1, 2, 3], enum_labels=["a", "b", "c"]
        )
        result = spec.to_dict()
        assert result["enum_values"] == [1, 2, 3]
        assert result["enum_labels"] == ["a", "b", "c"]

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "id": "test",
            "type": "uint32",
            "description": "Test property",
            "enum_values": [1, 2, 3],
            "enum_labels": ["a", "b", "c"],
        }
        spec = PropertySpec.from_dict(data)
        assert spec.id == "test"
        assert spec.type == "uint32"
        assert spec.description == "Test property"
        assert spec.enum_values == [1, 2, 3]
        assert spec.enum_labels == ["a", "b", "c"]

    def test_is_numeric(self):
        """Test numeric type checking."""
        spec = PropertySpec(id="test", type="uint32")
        assert spec.is_numeric() is True

        spec = PropertySpec(id="test", type="rgb")
        assert spec.is_numeric() is False

    def test_is_color(self):
        """Test color type checking."""
        spec = PropertySpec(id="test", type="rgb")
        assert spec.is_color() is True

        spec = PropertySpec(id="test", type="rgba")
        assert spec.is_color() is True

        spec = PropertySpec(id="test", type="uint32")
        assert spec.is_color() is False

    def test_has_enums(self):
        """Test enum checking."""
        spec = PropertySpec(id="test", type="uint32")
        assert spec.has_enums() is False

        spec = PropertySpec(id="test", type="uint32", enum_values=[1, 2], enum_labels=["a", "b"])
        assert spec.has_enums() is True

    def test_to_json(self):
        """Test JSON serialization."""
        spec = PropertySpec(id="test", type="uint32")
        json_str = spec.to_json()
        parsed = json.loads(json_str)
        assert parsed["id"] == "test"
        assert parsed["type"] == "uint32"

    def test_from_json(self):
        """Test creation from JSON string."""
        json_str = '{"id": "test", "type": "uint32"}'
        spec = PropertySpec.from_json(json_str)
        assert spec.id == "test"
        assert spec.type == "uint32"


class TestRelationship:
    """Test Relationship class."""

    def test_initialization_with_key(self):
        """Test initialization with explicit key."""
        rel = Relationship(id="Test Relationship", key="custom_key")
        assert rel.id == "Test Relationship"
        assert rel.key == "custom_key"

    def test_initialization_auto_key_generation(self):
        """Test initialization with automatic key generation."""
        rel = Relationship(id="Test Relationship!")
        assert rel.id == "Test Relationship!"
        assert rel.key == "test_relationship"

    def test_generate_key_from_id(self):
        """Test key generation from ID."""
        # Test with spaces and punctuation
        key = Relationship._generate_key_from_id("Test Relationship!")
        assert key == "test_relationship"

        # Test with mixed case and underscores (underscores are punctuation, get removed)
        key = Relationship._generate_key_from_id("My_Test-Property?")
        assert key == "mytestproperty"

        # Test with multiple spaces
        key = Relationship._generate_key_from_id("Multiple   Spaces")
        assert key == "multiple_spaces"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        rel = Relationship(id="test", key="test_key")
        result = rel.to_dict()
        expected = {"id": "test", "key": "test_key"}
        assert result == expected

    def test_to_dict_with_sharding(self):
        """Test conversion to dictionary with sharding."""
        sharding = ShardingSpec()
        rel = Relationship(id="test", key="test_key", sharding=sharding)
        result = rel.to_dict()
        assert "sharding" in result
        assert result["sharding"]["@type"] == "neuroglancer_uint64_sharded_v1"

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {"id": "test", "key": "test_key"}
        rel = Relationship.from_dict(data)
        assert rel.id == "test"
        assert rel.key == "test_key"

    def test_from_dict_without_key(self):
        """Test creation from dictionary without key (auto-generated)."""
        data = {"id": "Test Relationship"}
        rel = Relationship.from_dict(data)
        assert rel.id == "Test Relationship"
        assert rel.key == "test_relationship"

    def test_to_json(self):
        """Test JSON serialization."""
        rel = Relationship(id="test", key="test_key")
        json_str = rel.to_json()
        parsed = json.loads(json_str)
        assert parsed["id"] == "test"
        assert parsed["key"] == "test_key"

    def test_from_json(self):
        """Test creation from JSON string."""
        json_str = '{"id": "test", "key": "test_key"}'
        rel = Relationship.from_json(json_str)
        assert rel.id == "test"
        assert rel.key == "test_key"


class TestPointAnnotation:
    """Test PointAnnotation class."""

    def test_default_initialization(self):
        """Test default initialization."""
        point = PointAnnotation()
        assert point.position == (0.0, 0.0, 0.0)
        assert point.properties == {}
        assert point.relations == {}
        assert isinstance(point.id, int)

    def test_initialization_with_parameters(self):
        """Test initialization with parameters."""
        point = PointAnnotation(
            id=123,
            position=(1.0, 2.0, 3.0),
            properties={"color": [255, 0, 0]},
            relations={"synapse": [456, 789]},
        )
        assert point.id == 123
        assert point.position == (1.0, 2.0, 3.0)
        assert point.properties == {"color": [255, 0, 0]}
        assert point.relations == {"synapse": [456, 789]}

    def test_geometry_byte_size(self):
        """Test geometry byte size constant."""
        point = PointAnnotation()
        assert point.get_geometry_byte_size() == 12
        assert point.GEOMETRY_BYTES == 12
        assert PointAnnotation.GEOMETRY_BYTES == 12

    def test_write_geometry(self):
        """Test writing geometry data."""
        point = PointAnnotation(position=(1.5, 2.5, 3.5))
        output = io.BytesIO()
        point.write_geometry(output)

        output.seek(0)
        data = output.read()
        assert len(data) == 12

        # Verify the data
        unpacked = struct.unpack("<3f", data)
        assert unpacked == (1.5, 2.5, 3.5)

    def test_read_geometry(self):
        """Test reading geometry data."""
        # Create binary data
        data = struct.pack("<3f", 1.5, 2.5, 3.5)
        input_stream = io.BytesIO(data)

        point = PointAnnotation.read_geometry(input_stream)
        assert isinstance(point, PointAnnotation)
        assert point.position == (1.5, 2.5, 3.5)

    def test_in_bounds_with_bbox3d(self):
        """Test in_bounds with BBox3D."""
        point = PointAnnotation(position=(10.0, 20.0, 30.0))
        bbox = BBox3D.from_coords(
            start_coord=(5, 15, 25), end_coord=(15, 25, 35), resolution=(1, 1, 1)
        )

        assert point.in_bounds(bbox) is True

        # Point outside bounds
        point_outside = PointAnnotation(position=(0.0, 0.0, 0.0))
        assert point_outside.in_bounds(bbox) is False

    def test_in_bounds_with_default_resolution(self, mocker):
        """Test in_bounds with default resolution."""
        point = PointAnnotation(position=(10.0, 20.0, 30.0))

        # Mock bounds object without resolution attribute
        mock_bounds = mocker.Mock()
        mock_bounds.contains.return_value = True

        result = point.in_bounds(mock_bounds)  # No resolution specified
        assert result is True

        # Verify contains was called with position and default resolution
        mock_bounds.contains.assert_called_once_with((10.0, 20.0, 30.0), [1, 1, 1])

    def test_in_bounds_with_volumetric_index(self, mocker):
        """Test in_bounds with VolumetricIndex-like object."""
        point = PointAnnotation(position=(10.0, 20.0, 30.0))

        # Mock volumetric index
        mock_index = mocker.Mock()
        mock_index.resolution = Vec3D(2, 2, 2)
        mock_index.contains.return_value = True

        result = point.in_bounds(mock_index, resolution=(1, 1, 1))
        assert result is True

        # Verify the coordinates were converted
        mock_index.contains.assert_called_once_with((5.0, 10.0, 15.0))

    def test_in_bounds_with_generic_bounds(self, mocker):
        """Test in_bounds with generic bounds object."""
        point = PointAnnotation(position=(10.0, 20.0, 30.0))

        mock_bounds = mocker.Mock()
        mock_bounds.contains.return_value = True
        # No resolution attribute

        result = point.in_bounds(mock_bounds, resolution=(2, 2, 2))
        assert result is True

        mock_bounds.contains.assert_called_once_with((10.0, 20.0, 30.0), (2, 2, 2))

    def test_convert_coordinates(self):
        """Test mutable coordinate conversion."""
        point = PointAnnotation(position=(10.0, 20.0, 30.0))
        from_res = Vec3D(1, 1, 1)
        to_res = Vec3D(2, 2, 2)

        point.convert_coordinates(from_res, to_res)
        assert point.position == (5.0, 10.0, 15.0)

    def test_with_converted_coordinates(self):
        """Test immutable coordinate conversion."""
        point = PointAnnotation(
            id=123, position=(10.0, 20.0, 30.0), properties={"test": 1}, relations={"rel": [456]}
        )
        from_res = Vec3D(1, 1, 1)
        to_res = Vec3D(2, 2, 2)

        new_point = point.with_converted_coordinates(from_res, to_res)

        # Original unchanged
        assert point.position == (10.0, 20.0, 30.0)

        # New point has converted coordinates but same other data
        assert new_point.position == (5.0, 10.0, 15.0)
        assert new_point.id == 123
        assert new_point.properties == {"test": 1}
        assert new_point.relations == {"rel": [456]}


class TestLineAnnotation:
    """Test LineAnnotation class."""

    def test_default_initialization(self):
        """Test default initialization."""
        line = LineAnnotation()
        assert line.start == (0.0, 0.0, 0.0)
        assert line.end == (0.0, 0.0, 0.0)
        assert line.properties == {}
        assert line.relations == {}
        assert isinstance(line.id, int)

    def test_initialization_with_parameters(self):
        """Test initialization with parameters."""
        line = LineAnnotation(
            id=123,
            start=(1.0, 2.0, 3.0),
            end=(4.0, 5.0, 6.0),
            properties={"width": 2.5},
            relations={"synapse": [456]},
        )
        assert line.id == 123
        assert line.start == (1.0, 2.0, 3.0)
        assert line.end == (4.0, 5.0, 6.0)
        assert line.properties == {"width": 2.5}
        assert line.relations == {"synapse": [456]}

    def test_geometry_byte_size(self):
        """Test geometry byte size constant."""
        line = LineAnnotation()
        assert line.get_geometry_byte_size() == 24
        assert line.GEOMETRY_BYTES == 24
        assert LineAnnotation.GEOMETRY_BYTES == 24

    def test_write_geometry(self):
        """Test writing geometry data."""
        line = LineAnnotation(start=(1.5, 2.5, 3.5), end=(4.5, 5.5, 6.5))
        output = io.BytesIO()
        line.write_geometry(output)

        output.seek(0)
        data = output.read()
        assert len(data) == 24

        # Verify the data
        start_data = struct.unpack("<3f", data[:12])
        end_data = struct.unpack("<3f", data[12:])
        assert start_data == (1.5, 2.5, 3.5)
        assert end_data == (4.5, 5.5, 6.5)

    def test_read_geometry(self):
        """Test reading geometry data."""
        # Create binary data
        data = struct.pack("<3f", 1.5, 2.5, 3.5) + struct.pack("<3f", 4.5, 5.5, 6.5)
        input_stream = io.BytesIO(data)

        line = LineAnnotation.read_geometry(input_stream)
        assert isinstance(line, LineAnnotation)
        assert line.start == (1.5, 2.5, 3.5)
        assert line.end == (4.5, 5.5, 6.5)

    def test_in_bounds_strict_mode(self, mocker):
        """Test in_bounds with strict mode."""
        line = LineAnnotation(start=(10.0, 10.0, 10.0), end=(20.0, 20.0, 20.0))

        mock_bounds = mocker.Mock()
        mock_bounds.contains.side_effect = [True, True]  # Both endpoints in bounds

        result = line.in_bounds(mock_bounds, strict=True)
        assert result is True

        # Test with one endpoint outside
        mock_bounds.contains.side_effect = [True, False]
        result = line.in_bounds(mock_bounds, strict=True)
        assert result is False

    def test_in_bounds_non_strict_with_bbox3d(self, mocker):
        """Test in_bounds non-strict mode with BBox3D."""
        line = LineAnnotation(start=(0.0, 0.0, 0.0), end=(20.0, 20.0, 20.0))

        mock_bbox = mocker.Mock(spec=BBox3D)
        mock_bbox.line_intersects.return_value = True

        result = line.in_bounds(mock_bbox, strict=False)
        assert result is True

        mock_bbox.line_intersects.assert_called_once_with(
            (0.0, 0.0, 0.0), (20.0, 20.0, 20.0), resolution=(1, 1, 1)
        )

    def test_in_bounds_non_strict_with_volumetric_index(self, mocker):
        """Test in_bounds non-strict mode with VolumetricIndex-like object."""
        line = LineAnnotation(start=(10.0, 10.0, 10.0), end=(20.0, 20.0, 20.0))

        mock_index = mocker.Mock()
        mock_index.resolution = Vec3D(2, 2, 2)
        mock_index.line_intersects.return_value = True

        result = line.in_bounds(mock_index, resolution=(1, 1, 1), strict=False)
        assert result is True

        # Verify coordinates were converted
        mock_index.line_intersects.assert_called_once_with((5.0, 5.0, 5.0), (10.0, 10.0, 10.0))

    def test_in_bounds_with_default_resolution(self):
        """Test in_bounds with default resolution."""
        line = LineAnnotation(start=(5.0, 5.0, 5.0), end=(15.0, 15.0, 15.0))
        bbox = BBox3D.from_coords(
            start_coord=(0, 0, 0), end_coord=(20, 20, 20), resolution=(1, 1, 1)
        )

        result = line.in_bounds(bbox)  # No resolution specified
        assert result is True

    def test_convert_coordinates(self):
        """Test mutable coordinate conversion."""
        line = LineAnnotation(start=(10.0, 20.0, 30.0), end=(40.0, 50.0, 60.0))
        from_res = Vec3D(1, 1, 1)
        to_res = Vec3D(2, 2, 2)

        line.convert_coordinates(from_res, to_res)
        assert line.start == (5.0, 10.0, 15.0)
        assert line.end == (20.0, 25.0, 30.0)

    def test_with_converted_coordinates(self):
        """Test immutable coordinate conversion."""
        line = LineAnnotation(
            id=123,
            start=(10.0, 20.0, 30.0),
            end=(40.0, 50.0, 60.0),
            properties={"test": 1},
            relations={"rel": [456]},
        )
        from_res = Vec3D(1, 1, 1)
        to_res = Vec3D(2, 2, 2)

        new_line = line.with_converted_coordinates(from_res, to_res)

        # Original unchanged
        assert line.start == (10.0, 20.0, 30.0)
        assert line.end == (40.0, 50.0, 60.0)

        # New line has converted coordinates but same other data
        assert new_line.start == (5.0, 10.0, 15.0)
        assert new_line.end == (20.0, 25.0, 30.0)
        assert new_line.id == 123
        assert new_line.properties == {"test": 1}
        assert new_line.relations == {"rel": [456]}


class TestAnnotationPropertyMethods:
    """Test property read/write methods for annotations."""

    def test_write_properties_4byte_types(self):
        """Test writing 4-byte properties."""
        annotation = PointAnnotation(
            properties={"uint32_prop": 42, "int32_prop": -42, "float32_prop": 3.14}
        )

        specs = [
            PropertySpec(id="uint32_prop", type="uint32"),
            PropertySpec(id="int32_prop", type="int32"),
            PropertySpec(id="float32_prop", type="float32"),
        ]

        output = io.BytesIO()
        annotation.write_properties(output, specs)

        output.seek(0)
        data = output.read()
        assert len(data) == 12  # 3 * 4 bytes

        # Verify data
        uint32_val = struct.unpack("<I", data[0:4])[0]
        int32_val = struct.unpack("<i", data[4:8])[0]
        float32_val = struct.unpack("<f", data[8:12])[0]

        assert uint32_val == 42
        assert int32_val == -42
        assert abs(float32_val - 3.14) < 0.001

    def test_write_properties_2byte_types(self):
        """Test writing 2-byte properties."""
        annotation = PointAnnotation(properties={"uint16_prop": 1000, "int16_prop": -1000})

        specs = [
            PropertySpec(id="uint16_prop", type="uint16"),
            PropertySpec(id="int16_prop", type="int16"),
        ]

        output = io.BytesIO()
        annotation.write_properties(output, specs)

        output.seek(0)
        data = output.read()
        # 2 * 2 bytes + 2 bytes padding = 6 bytes
        assert len(data) == 8  # Rounded to 4-byte alignment

        # Verify data
        uint16_val = struct.unpack("<H", data[0:2])[0]
        int16_val = struct.unpack("<h", data[2:4])[0]

        assert uint16_val == 1000
        assert int16_val == -1000

    def test_write_properties_1byte_and_color_types(self):
        """Test writing 1-byte and color properties."""
        annotation = PointAnnotation(
            properties={
                "uint8_prop": 255,
                "int8_prop": -128,
                "rgb_prop": [255, 128, 64],
                "rgba_prop": [255, 128, 64, 32],
            }
        )

        specs = [
            PropertySpec(id="uint8_prop", type="uint8"),
            PropertySpec(id="int8_prop", type="int8"),
            PropertySpec(id="rgb_prop", type="rgb"),
            PropertySpec(id="rgba_prop", type="rgba"),
        ]

        output = io.BytesIO()
        annotation.write_properties(output, specs)

        output.seek(0)
        data = output.read()
        # 1 + 1 + 3 + 4 = 9 bytes + 3 bytes padding = 12 bytes
        assert len(data) == 12

        # Verify data
        uint8_val = struct.unpack("<B", data[0:1])[0]
        int8_val = struct.unpack("<b", data[1:2])[0]
        rgb_vals = struct.unpack("<3B", data[2:5])
        rgba_vals = struct.unpack("<4B", data[5:9])

        assert uint8_val == 255
        assert int8_val == -128
        assert rgb_vals == (255, 128, 64)
        assert rgba_vals == (255, 128, 64, 32)

    def test_write_properties_missing_values_use_defaults(self):
        """Test that missing property values use defaults."""
        annotation = PointAnnotation(properties={})  # No properties set

        specs = [
            PropertySpec(id="missing_uint32", type="uint32"),
            PropertySpec(id="missing_rgb", type="rgb"),
            PropertySpec(id="missing_rgba", type="rgba"),
        ]

        output = io.BytesIO()
        annotation.write_properties(output, specs)

        output.seek(0)
        data = output.read()

        # Verify defaults are used
        uint32_val = struct.unpack("<I", data[0:4])[0]
        rgb_vals = struct.unpack("<3B", data[4:7])
        rgba_vals = struct.unpack("<4B", data[7:11])

        assert uint32_val == 0
        assert rgb_vals == (0, 0, 0)
        assert rgba_vals == (0, 0, 0, 255)

    def test_read_properties_4byte_types(self):
        """Test reading 4-byte properties."""
        data = struct.pack("<I", 42) + struct.pack("<i", -42) + struct.pack("<f", 3.14)
        input_stream = io.BytesIO(data)

        specs = [
            PropertySpec(id="uint32_prop", type="uint32"),
            PropertySpec(id="int32_prop", type="int32"),
            PropertySpec(id="float32_prop", type="float32"),
        ]

        annotation = PointAnnotation()
        annotation.read_properties(input_stream, specs)

        assert annotation.properties["uint32_prop"] == 42
        assert annotation.properties["int32_prop"] == -42
        assert abs(annotation.properties["float32_prop"] - 3.14) < 0.001

    def test_read_properties_2byte_types(self):
        """Test reading 2-byte properties."""
        data = struct.pack("<H", 1000) + struct.pack("<h", -1000) + b"\x00\x00"  # padding
        input_stream = io.BytesIO(data)

        specs = [
            PropertySpec(id="uint16_prop", type="uint16"),
            PropertySpec(id="int16_prop", type="int16"),
        ]

        annotation = PointAnnotation()
        annotation.read_properties(input_stream, specs)

        assert annotation.properties["uint16_prop"] == 1000
        assert annotation.properties["int16_prop"] == -1000

    def test_read_properties_1byte_and_color_types(self):
        """Test reading 1-byte and color properties."""
        data = (
            struct.pack("<B", 255)
            + struct.pack("<b", -128)
            + struct.pack("<3B", 255, 128, 64)
            + struct.pack("<4B", 255, 128, 64, 32)
            + b"\x00\x00\x00"
        )  # padding to 4-byte alignment
        input_stream = io.BytesIO(data)

        specs = [
            PropertySpec(id="uint8_prop", type="uint8"),
            PropertySpec(id="int8_prop", type="int8"),
            PropertySpec(id="rgb_prop", type="rgb"),
            PropertySpec(id="rgba_prop", type="rgba"),
        ]

        annotation = PointAnnotation()
        annotation.read_properties(input_stream, specs)

        assert annotation.properties["uint8_prop"] == 255
        assert annotation.properties["int8_prop"] == -128
        assert annotation.properties["rgb_prop"] == [255, 128, 64]
        assert annotation.properties["rgba_prop"] == [255, 128, 64, 32]


class TestAnnotationWriteRead:
    """Test complete annotation write/read cycle."""

    def test_write_read_point_minimal(self):
        """Test writing and reading a minimal point annotation."""
        point = PointAnnotation(id=123, position=(1.5, 2.5, 3.5))

        output = io.BytesIO()
        point.write(output)

        output.seek(0)
        data = output.read()
        assert len(data) == 12  # Just geometry

        # Read it back
        input_stream = io.BytesIO(data)
        read_point = Annotation.read(input_stream, "POINT")
        assert isinstance(read_point, PointAnnotation)
        assert read_point.position == (1.5, 2.5, 3.5)
        assert read_point.properties == {}
        assert read_point.relations == {}

    def test_write_read_line_minimal(self):
        """Test writing and reading a minimal line annotation."""
        line = LineAnnotation(id=123, start=(1.5, 2.5, 3.5), end=(4.5, 5.5, 6.5))

        output = io.BytesIO()
        line.write(output)

        output.seek(0)
        data = output.read()
        assert len(data) == 24  # Just geometry

        # Read it back
        input_stream = io.BytesIO(data)
        read_line = Annotation.read(input_stream, "LINE")
        assert isinstance(read_line, LineAnnotation)
        assert read_line.start == (1.5, 2.5, 3.5)
        assert read_line.end == (4.5, 5.5, 6.5)
        assert read_line.properties == {}
        assert read_line.relations == {}

    def test_write_read_point_with_properties(self):
        """Test writing and reading point annotation with properties."""
        point = PointAnnotation(
            id=123, position=(1.5, 2.5, 3.5), properties={"size": 10, "color": [255, 128, 64]}
        )

        specs = [
            PropertySpec(id="size", type="uint32"),
            PropertySpec(id="color", type="rgb"),
        ]

        output = io.BytesIO()
        point.write(output, property_specs=specs)

        output.seek(0)

        # Read it back
        read_point = Annotation.read(output, "POINT", property_specs=specs)
        assert isinstance(read_point, PointAnnotation)
        assert read_point.position == (1.5, 2.5, 3.5)
        assert read_point.properties["size"] == 10
        assert read_point.properties["color"] == [255, 128, 64]

    def test_write_read_point_with_relationships(self):
        """Test writing and reading point annotation with relationships."""
        point = PointAnnotation(
            id=123, position=(1.5, 2.5, 3.5), relations={"synapse": [456, 789], "dendrite": [999]}
        )

        relationships = [
            Relationship(id="synapse"),
            Relationship(id="dendrite"),
        ]

        output = io.BytesIO()
        point.write(output, relationships=relationships)

        output.seek(0)

        # Read it back
        read_point = Annotation.read(output, "POINT", relationships=relationships)
        assert isinstance(read_point, PointAnnotation)
        assert read_point.position == (1.5, 2.5, 3.5)
        assert read_point.relations["synapse"] == [456, 789]
        assert read_point.relations["dendrite"] == [999]

    def test_write_read_with_scalar_relation(self):
        """Test writing and reading annotation with scalar relation value."""
        point = PointAnnotation(
            id=123, position=(1.5, 2.5, 3.5), relations={"parent": 456}  # Scalar value
        )

        relationships = [Relationship(id="parent")]

        output = io.BytesIO()
        point.write(output, relationships=relationships)

        output.seek(0)

        # Read it back
        read_point = Annotation.read(output, "POINT", relationships=relationships)
        assert isinstance(read_point, PointAnnotation)
        assert read_point.relations["parent"] == [456]  # Should be converted to list

    def test_read_invalid_type(self):
        """Test reading with invalid annotation type."""
        data = struct.pack("<3f", 1.0, 2.0, 3.0)
        input_stream = io.BytesIO(data)

        with pytest.raises(ValueError, match="type: expected POINT or LINE, but got 'INVALID'"):
            Annotation.read(input_stream, "INVALID")


class TestSpatialEntry:
    """Test SpatialEntry class."""

    def test_initialization(self):
        """Test SpatialEntry initialization."""
        entry = SpatialEntry(
            chunk_size=[64, 64, 32], grid_shape=[10, 10, 5], key="spatial0", limit=1000
        )

        assert entry.chunk_size == (64, 64, 32)  # Converted to tuple
        assert entry.grid_shape == (10, 10, 5)  # Converted to tuple
        assert entry.key == "spatial0"
        assert entry.limit == 1000
        assert entry.sharding is None

    def test_initialization_with_sharding(self):
        """Test SpatialEntry initialization with sharding."""
        sharding = ShardingSpec()
        entry = SpatialEntry(
            chunk_size=[64, 64, 32],
            grid_shape=[10, 10, 5],
            key="spatial0",
            limit=1000,
            sharding=sharding,
        )

        assert entry.sharding is sharding

    def test_to_json_without_sharding(self):
        """Test JSON serialization without sharding."""
        entry = SpatialEntry(
            chunk_size=[64, 64, 32], grid_shape=[10, 10, 5], key="spatial0", limit=1000
        )

        json_str = entry.to_json()
        parsed = json.loads(json_str)

        assert parsed["chunk_size"] == [64, 64, 32]
        assert parsed["grid_shape"] == [10, 10, 5]
        assert parsed["key"] == "spatial0"
        assert parsed["limit"] == 1000

    def test_to_json_with_sharding(self):
        """Test JSON serialization with sharding."""
        sharding = ShardingSpec()
        entry = SpatialEntry(
            chunk_size=[64, 64, 32],
            grid_shape=[10, 10, 5],
            key="spatial0",
            limit=1000,
            sharding=sharding,
        )

        json_str = entry.to_json()
        parsed = json.loads(json_str)

        assert "sharding" in parsed
        assert parsed["sharding"]["@type"] == "neuroglancer_uint64_sharded_v1"


class TestValidateSpatialEntries:
    """Test validate_spatial_entries function."""

    def test_empty_list(self):
        """Test validation with empty list."""
        assert validate_spatial_entries([]) is True

    def test_single_entry(self):
        """Test validation with single entry."""
        entry = SpatialEntry(
            chunk_size=[64, 64, 32], grid_shape=[10, 10, 5], key="spatial0", limit=1000
        )
        assert validate_spatial_entries([entry]) is True

    def test_valid_two_level_hierarchy(self):
        """Test validation with valid two-level hierarchy."""
        entry1 = SpatialEntry(
            chunk_size=[64, 64, 32], grid_shape=[10, 10, 5], key="spatial0", limit=1000
        )
        entry2 = SpatialEntry(
            chunk_size=[32, 32, 16],  # Smaller chunks
            grid_shape=[20, 20, 10],  # More chunks, same total size
            key="spatial1",
            limit=500,
        )

        assert validate_spatial_entries([entry1, entry2]) is True

    def test_invalid_chunk_size_increase(self):
        """Test validation fails when chunk size increases."""
        entry1 = SpatialEntry(
            chunk_size=[32, 32, 16], grid_shape=[10, 10, 5], key="spatial0", limit=1000
        )
        entry2 = SpatialEntry(
            chunk_size=[64, 32, 16],  # Chunk size increased in first dimension
            grid_shape=[5, 10, 5],
            key="spatial1",
            limit=500,
        )

        with pytest.raises(ValueError, match="chunk_size.*is larger than previous"):
            validate_spatial_entries([entry1, entry2])

    def test_invalid_total_size_mismatch(self):
        """Test validation fails when total size changes."""
        entry1 = SpatialEntry(
            chunk_size=[64, 64, 32],
            grid_shape=[10, 10, 5],  # Total: 640x640x160
            key="spatial0",
            limit=1000,
        )
        entry2 = SpatialEntry(
            chunk_size=[32, 32, 16],
            grid_shape=[20, 20, 20],  # Total: 640x640x320 (different Z size)
            key="spatial1",
            limit=500,
        )

        with pytest.raises(ValueError, match="total size.*does not match"):
            validate_spatial_entries([entry1, entry2])

    def test_valid_three_level_hierarchy(self):
        """Test validation with valid three-level hierarchy."""
        entry1 = SpatialEntry(
            chunk_size=[128, 128, 64], grid_shape=[8, 8, 4], key="spatial0", limit=2000
        )
        entry2 = SpatialEntry(
            chunk_size=[64, 64, 32], grid_shape=[16, 16, 8], key="spatial1", limit=1000
        )
        entry3 = SpatialEntry(
            chunk_size=[32, 32, 16], grid_shape=[32, 32, 16], key="spatial2", limit=500
        )

        assert validate_spatial_entries([entry1, entry2, entry3]) is True


class TestGetChildCellRanges:
    """Test get_child_cell_ranges function."""

    def test_simple_subdivision(self):
        """Test simple 2:1 subdivision in all dimensions."""
        parent_entry = SpatialEntry(
            chunk_size=[64, 64, 32],
            grid_shape=[4, 4, 2],  # 4x4x2 parent cells
            key="spatial0",
            limit=1000,
        )
        child_entry = SpatialEntry(
            chunk_size=[32, 32, 16],
            grid_shape=[8, 8, 4],  # 8x8x4 child cells (2x subdivision)
            key="spatial1",
            limit=500,
        )

        spatial_specs = [parent_entry, child_entry]

        # Test parent cell (0, 0, 0)
        ranges = get_child_cell_ranges(spatial_specs, 0, (0, 0, 0))
        assert ranges == ((0, 2), (0, 2), (0, 2))

        # Test parent cell (1, 1, 1)
        ranges = get_child_cell_ranges(spatial_specs, 0, (1, 1, 1))
        assert ranges == ((2, 4), (2, 4), (2, 4))

        # Test parent cell (3, 3, 1) - last parent cell
        ranges = get_child_cell_ranges(spatial_specs, 0, (3, 3, 1))
        assert ranges == ((6, 8), (6, 8), (2, 4))

    def test_non_uniform_subdivision(self):
        """Test non-uniform subdivision across dimensions."""
        parent_entry = SpatialEntry(
            chunk_size=[128, 64, 32],
            grid_shape=[2, 4, 2],  # 2x4x2 parent cells
            key="spatial0",
            limit=1000,
        )
        child_entry = SpatialEntry(
            chunk_size=[64, 32, 16],
            grid_shape=[4, 8, 4],  # 4x8x4 child cells (2x, 2x, 2x subdivision)
            key="spatial1",
            limit=500,
        )

        spatial_specs = [parent_entry, child_entry]

        # Test parent cell (0, 0, 0)
        ranges = get_child_cell_ranges(spatial_specs, 0, (0, 0, 0))
        assert ranges == ((0, 2), (0, 2), (0, 2))

        # Test parent cell (1, 3, 1)
        ranges = get_child_cell_ranges(spatial_specs, 0, (1, 3, 1))
        assert ranges == ((2, 4), (6, 8), (2, 4))

    def test_uneven_subdivision(self):
        """Test subdivision where child grid is not evenly divisible."""
        parent_entry = SpatialEntry(
            chunk_size=[96, 96, 48],
            grid_shape=[3, 3, 3],  # 3x3x3 parent cells
            key="spatial0",
            limit=1000,
        )
        child_entry = SpatialEntry(
            chunk_size=[32, 32, 16],
            grid_shape=[9, 9, 9],  # 9x9x9 child cells (3x subdivision)
            key="spatial1",
            limit=500,
        )

        spatial_specs = [parent_entry, child_entry]

        # Test parent cell (0, 0, 0)
        ranges = get_child_cell_ranges(spatial_specs, 0, (0, 0, 0))
        assert ranges == ((0, 3), (0, 3), (0, 3))

        # Test parent cell (1, 1, 1)
        ranges = get_child_cell_ranges(spatial_specs, 0, (1, 1, 1))
        assert ranges == ((3, 6), (3, 6), (3, 6))

        # Test parent cell (2, 2, 2) - last parent cell
        ranges = get_child_cell_ranges(spatial_specs, 0, (2, 2, 2))
        assert ranges == ((6, 9), (6, 9), (6, 9))
