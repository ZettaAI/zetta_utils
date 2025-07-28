"""
Data classes and support code for Neuroglancer annotations (primarily
in precomputed format).  Reference:

https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/annotations.md
https://github.com/google/neuroglancer/issues/227#issuecomment-2246350747
"""

import json
import random
import re
import string
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import IO, Any, ClassVar, Literal, Sequence

import numpy as np

from zetta_utils.geometry import Vec3D
from zetta_utils.geometry.bbox import BBox3D
from zetta_utils.geometry.vec import VEC3D_PRECISION


@dataclass
class ShardingSpec:
    """
    Represents a Neuroglancer sharding specification.

    Based on the neuroglancer_uint64_sharded_v1 format.
    """

    preshift_bits: int = 0
    hash: Literal["identity", "murmurhash3_x86_128"] = "identity"
    minishard_bits: int = 3
    shard_bits: int = 4
    minishard_index_encoding: Literal["raw", "gzip"] | None = "raw"
    data_encoding: Literal["raw", "gzip"] | None = "raw"

    @property
    def type(self) -> str:
        """Always returns the required @type value."""
        return "neuroglancer_uint64_sharded_v1"

    @property
    def num_shards(self) -> int:
        """Number of shards: 2**shard_bits"""
        return 2 ** self.shard_bits

    @property
    def num_minishards_per_shard(self) -> int:
        """Number of minishards per shard: 2**minishard_bits"""
        return 2 ** self.minishard_bits

    def get_shard_number(self, chunk_id: int) -> int:
        """
        Compute the shard number for a given chunk_id.

        :param chunk_id: The chunk identifier (uint64)
        :return: Shard number in range [0, num_shards)
        """
        if self.hash == "identity":
            hashed_chunk_id = chunk_id >> self.preshift_bits
        elif self.hash == "murmurhash3_x86_128":  # pragma: no cover
            # Would need to implement MurmurHash3_x86_128 here
            raise NotImplementedError("MurmurHash3_x86_128 not implemented")
        else:  # pragma: no cover
            raise ValueError(f"Unknown hash function: {self.hash}")

        # Extract shard bits: [minishard_bits, minishard_bits+shard_bits)
        shard_number = (hashed_chunk_id >> self.minishard_bits) & ((1 << self.shard_bits) - 1)
        return shard_number

    def get_minishard_number(self, chunk_id: int) -> int:
        """
        Compute the minishard number for a given chunk_id.

        :param chunk_id: The chunk identifier (uint64)
        :return: Minishard number in range [0, num_minishards_per_shard)
        """
        if self.hash == "identity":
            hashed_chunk_id = chunk_id >> self.preshift_bits
        elif self.hash == "murmurhash3_x86_128":  # pragma: no cover
            # Would need to implement MurmurHash3_x86_128 here
            raise NotImplementedError("MurmurHash3_x86_128 not implemented")
        else:  # pragma: no cover
            raise ValueError(f"Unknown hash function: {self.hash}")

        # Extract minishard bits: [0, minishard_bits)
        minishard_number = hashed_chunk_id & ((1 << self.minishard_bits) - 1)
        return minishard_number

    def get_shard_filename(self, shard_number: int) -> str:
        """
        Get the filename for a given shard number.

        :param shard_number: Shard number in range [0, num_shards)
        :return: Filename in format: {shard_number:0{width}x}.shard
        """
        if not 0 <= shard_number < self.num_shards:
            raise ValueError(f"Shard number {shard_number} out of range [0, {self.num_shards})")

        # Zero-pad to ceil(shard_bits/4) digits
        width = (self.shard_bits + 3) // 4  # equivalent to ceil(shard_bits/4)
        return f"{shard_number:0{width}x}.shard"

    def to_dict(self) -> dict:
        """Convert to dictionary format matching the JSON specification."""
        result = {
            "@type": self.type,
            "preshift_bits": self.preshift_bits,
            "hash": self.hash,
            "minishard_bits": self.minishard_bits,
            "shard_bits": self.shard_bits,
        }

        if self.minishard_index_encoding != "raw":
            result["minishard_index_encoding"] = self.minishard_index_encoding  # pragma: no cover

        if self.data_encoding != "raw":
            result["data_encoding"] = self.data_encoding  # pragma: no cover

        return result

    def to_json(self, indent: int | None = None) -> str:
        """Convert to JSON string representation.

        :param indent: number of spaces for indentation; None for compact JSON
        :return: JSON string representation of the property specification.
        """
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class PropertySpec:
    """Represents a property definition for Neuroglancer precomputed annotations.

    Based on the specification at:
    https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/annotations.md
    """

    # Valid property types as defined in the spec
    VALID_TYPES = {"rgb", "rgba", "uint8", "int8", "uint16", "int16", "uint32", "int32", "float32"}

    # Numeric types that support enum values
    NUMERIC_TYPES = {"uint8", "int8", "uint16", "int16", "uint32", "int32", "float32"}

    id: str
    type: str
    description: str | None = None
    enum_values: list[int | float] | None = None
    enum_labels: list[str] | None = None

    def __post_init__(self):
        """Validate the property specification after initialization."""
        self._validate_id()
        self._validate_type()
        self._validate_enums()

    def _validate_id(self):
        """Validate that the id follows the required pattern."""
        if not re.match(r"^[a-z][a-zA-Z0-9_]*$", self.id):
            raise ValueError(
                f"Property id '{self.id}' must match pattern: "
                "start with lowercase letter, followed by letters, digits, or underscores"
            )  # pragma: no cover

    def _validate_type(self):
        """Validate that the type is one of the supported types."""
        if self.type not in self.VALID_TYPES:
            raise ValueError(
                f"Property type '{self.type}' must be one of: "
                + ", ".join(sorted(self.VALID_TYPES))
            )  # pragma: no cover

    def _validate_enums(self):
        """Validate enum_values and enum_labels consistency."""
        has_values = self.enum_values is not None
        has_labels = self.enum_labels is not None

        # Both must be provided together or not at all
        if has_values != has_labels:
            raise ValueError(
                "enum_values and enum_labels must both be provided or both be None"
            )  # pragma: no cover

        # If enums are provided, validate additional constraints
        if has_values:
            # Only numeric types support enums
            if self.type not in self.NUMERIC_TYPES:
                raise ValueError(
                    f"enum_values not supported for type '{self.type}'. "
                    f"Only numeric types support enums: {', '.join(sorted(self.NUMERIC_TYPES))}"
                )

            # Arrays must have same length
            assert self.enum_values is not None
            assert self.enum_labels is not None
            if len(self.enum_values) != len(self.enum_labels):
                raise ValueError(
                    f"enum_values and enum_labels must have same length: "
                    f"{len(self.enum_values)} vs {len(self.enum_labels)}"
                )  # pragma: no cover

    def to_dict(self) -> dict:
        """Convert to dictionary format matching the JSON specification."""
        result: dict[str, Any] = {"id": self.id, "type": self.type}

        if self.description is not None:
            result["description"] = self.description

        if self.enum_values is not None:
            result["enum_values"] = self.enum_values
            result["enum_labels"] = self.enum_labels

        return result

    @classmethod
    def from_dict(cls, data: dict) -> "PropertySpec":
        """Create PropertySpec from dictionary."""
        return cls(
            id=data["id"],
            type=data["type"],
            description=data.get("description"),
            enum_values=data.get("enum_values"),
            enum_labels=data.get("enum_labels"),
        )

    def is_numeric(self) -> bool:
        """Check if this property is a numeric type."""
        return self.type in self.NUMERIC_TYPES

    def is_color(self) -> bool:
        """Check if this property is a color type (rgb/rgba)."""
        return self.type in {"rgb", "rgba"}

    def has_enums(self) -> bool:
        """Check if this property has enumerated values."""
        return self.enum_values is not None

    def to_json(self, indent: int | None = None) -> str:
        """Convert to JSON string representation.

        :param indent: number of spaces for indentation; None for compact JSON
        :return: JSON string representation of the property specification.
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "PropertySpec":
        """Create PropertySpec from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class Relationship:
    """Represents a relationship definition for Neuroglancer precomputed annotations.

    Based on the specification at:
    https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/annotations.md
    """

    id: str
    key: str | None = None
    sharding: ShardingSpec | None = None

    def __post_init__(self):
        """Generate key from id if not provided."""
        if self.key is None:
            self.key = self._generate_key_from_id(self.id)

    @staticmethod
    def _generate_key_from_id(id_str: str) -> str:
        """Generate a key by lowercasing, removing punctuation, and replacing
        spaces with underscores.

        :param id_str: The ID string to convert
        :return: Cleaned key string suitable for use as a directory name
        """
        # Convert to lowercase
        key = id_str.lower()

        # Remove all punctuation
        key = key.translate(str.maketrans("", "", string.punctuation))

        # Replace spaces (and any other whitespace) with underscores
        key = re.sub(r"\s+", "_", key)

        return key

    def to_dict(self) -> dict:
        """Convert to dictionary format matching the JSON specification."""
        result: dict[str, Any] = {"id": self.id, "key": self.key}
        if self.sharding is not None:
            result["sharding"] = self.sharding.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "Relationship":
        """Create Relationship from dictionary."""
        return cls(id=data["id"], key=data.get("key"))  # Will auto-generate if None

    def to_json(self, indent: int | None = None) -> str:
        """Convert to JSON string representation.

        :param indent: Number of spaces for indentation. None for compact JSON.
        :return: JSON string representation of the relationship.
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "Relationship":
        """Create Relationship from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class Annotation(ABC):
    """Base class for all annotation types in Neuroglancer precomputed format.

    Based on the specification at:
    https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/annotations.md
    """

    id: int = field(default_factory=lambda: random.randint(0, 2 ** 64 - 1))
    properties: dict[str, Any] = field(default_factory=dict)
    relations: dict[str, int | list[int]] = field(default_factory=dict)

    @abstractmethod
    def in_bounds(
        self, bounds, resolution: Sequence[float] | None = None, strict: bool = False
    ) -> bool:
        """Return whether either end of this line is in the given bounds.

        :param bounds: VolumetricIndex or BBox3D
        :param resolution: Resolution of the coordinates in this annotation.
        :param strict: If true, return True only if entirely contained within the bounds.
        """

    @abstractmethod
    def write_geometry(self, output: IO[bytes]) -> None:
        """Write the geometry-specific data (positions/radii) in binary format.

        This method should write the position/radii vectors required by the
        annotation type as float32le values in the order specified by the spec.
        """

    @abstractmethod
    def get_geometry_byte_size(self) -> int:
        """Return the number of bytes used by the geometry data."""

    @classmethod
    @abstractmethod
    def read_geometry(cls, in_stream: IO[bytes]) -> "Annotation":
        """Read geometry-specific data from binary format.

        Should return a new instance with geometry data populated,
        but without properties or relationships.
        """

    @abstractmethod
    def convert_coordinates(self, from_res, to_res) -> None:
        """Convert coordinates from one resolution to another (mutates instance).

        :param from_res: Source resolution (Vec3D)
        :param to_res: Target resolution (Vec3D)
        """

    @abstractmethod
    def with_converted_coordinates(self, from_res, to_res) -> "Annotation":
        """Return new Annotation with converted coordinates.

        :param from_res: Source resolution (Vec3D)
        :param to_res: Target resolution (Vec3D)
        """

    def write_properties(
        self, output: IO[bytes], property_specs: Sequence["PropertySpec"]
    ) -> None:
        """Write property values in the order specified by property_specs.

        Properties are written in order: uint32/int32/float32, then uint16/int16,
        then uint8/int8/rgb/rgba, followed by padding to 4-byte alignment.
        """
        # pylint: disable=too-many-branches
        bytes_written = 0

        # Write 4-byte properties
        for spec in property_specs:
            value = self.properties.get(spec.id, 0)
            if spec.type == "uint32":
                output.write(struct.pack("<I", int(value)))
            elif spec.type == "int32":
                output.write(struct.pack("<i", int(value)))
            elif spec.type == "float32":
                output.write(struct.pack("<f", float(value)))
            else:
                continue
            bytes_written += 4

        # Write 2-byte properties
        for spec in property_specs:
            value = self.properties.get(spec.id, 0)
            if spec.type == "uint16":
                output.write(struct.pack("<H", int(value)))
            elif spec.type == "int16":
                output.write(struct.pack("<h", int(value)))
            else:
                continue
            bytes_written += 2

        # Write 1-byte properties and color properties
        for spec in property_specs:
            if spec.type in {"uint8", "int8"}:
                value = self.properties.get(spec.id, 0)
                if spec.type == "uint8":
                    output.write(struct.pack("<B", int(value)))
                else:  # int8
                    output.write(struct.pack("<b", int(value)))
                bytes_written += 1
            elif spec.type == "rgb":
                value = self.properties.get(spec.id, [0, 0, 0])
                output.write(struct.pack("<3B", *value[:3]))
                bytes_written += 3
            elif spec.type == "rgba":
                value = self.properties.get(spec.id, [0, 0, 0, 255])
                output.write(struct.pack("<4B", *value[:4]))
                bytes_written += 4

        # Add padding to reach 4-byte alignment
        padding_needed = (4 - (bytes_written % 4)) % 4
        if padding_needed > 0:
            output.write(b"\x00" * padding_needed)
        # pylint: enable=too-many-branches

    def read_properties(
        self, in_stream: IO[bytes], property_specs: Sequence["PropertySpec"]
    ) -> None:
        """Read property values in the order specified by property_specs.

        Properties are read in order: uint32/int32/float32, then uint16/int16,
        then uint8/int8/rgb/rgba, followed by padding to 4-byte alignment.
        """
        # pylint: disable=too-many-branches
        bytes_read = 0

        # Read 4-byte properties
        for spec in property_specs:
            if spec.type == "uint32":
                self.properties[spec.id] = struct.unpack("<I", in_stream.read(4))[0]
            elif spec.type == "int32":
                self.properties[spec.id] = struct.unpack("<i", in_stream.read(4))[0]
            elif spec.type == "float32":
                self.properties[spec.id] = struct.unpack("<f", in_stream.read(4))[0]
            else:
                continue
            bytes_read += 4

        # Read 2-byte properties
        for spec in property_specs:
            if spec.type == "uint16":
                self.properties[spec.id] = struct.unpack("<H", in_stream.read(2))[0]
            elif spec.type == "int16":
                self.properties[spec.id] = struct.unpack("<h", in_stream.read(2))[0]
            else:
                continue
            bytes_read += 2

        # Write 1-byte properties and color properties
        for spec in property_specs:
            if spec.type in {"uint8", "int8"}:
                if spec.type == "uint8":
                    self.properties[spec.id] = struct.unpack("<B", in_stream.read(1))[0]
                else:  # int8
                    self.properties[spec.id] = struct.unpack("<b", in_stream.read(1))[0]
                bytes_read += 1
            elif spec.type == "rgb":
                self.properties[spec.id] = list(struct.unpack("<3B", in_stream.read(3)))
                bytes_read += 3
            elif spec.type == "rgba":
                self.properties[spec.id] = list(struct.unpack("<4B", in_stream.read(4)))
                bytes_read += 4

        # Read padding to reach 4-byte alignment
        padding_needed = (4 - (bytes_read % 4)) % 4
        if padding_needed > 0:
            in_stream.read(padding_needed)
        # pylint: enable=too-many-branches

    def write(
        self,
        output: IO[bytes],
        property_specs: Sequence[PropertySpec] | None = None,
        relationships: Sequence[Relationship] | None = None,
    ) -> None:
        """Write complete annotation in binary format.

        :param output: Binary output stream
        :param property_specs: List of PropertySpec objects defining property order and types
        :param relationships: List of lists of related object IDs (for annotation ID index only)
        """
        # Write geometry data
        self.write_geometry(output)

        # Write properties if specified
        if property_specs:
            self.write_properties(output, property_specs)

        # Write relationships if specified
        # (Relationships should be given ONLY for by_id index)
        if relationships is not None:
            for rel in relationships:
                related_ids = self.relations.get(rel.id, [])
                if np.isscalar(related_ids):
                    related_ids = [related_ids]  # type: ignore
                assert isinstance(related_ids, (list, tuple))
                output.write(struct.pack("<I", len(related_ids)))
                for idnum in related_ids:
                    output.write(struct.pack("<Q", idnum))

    @classmethod
    def read(
        cls,
        in_stream: IO[bytes],
        type: str,  # pylint: disable=redefined-builtin
        property_specs: Sequence[PropertySpec] | None = None,
        relationships: Sequence[Relationship] | None = None,
    ) -> "Annotation":
        result: Annotation
        if type == "POINT":
            result = PointAnnotation.read_geometry(in_stream)
        elif type == "LINE":
            result = LineAnnotation.read_geometry(in_stream)
        else:
            raise ValueError(f"type: expected POINT or LINE, but got '{type}'")
        if property_specs:
            result.read_properties(in_stream, property_specs)
        if relationships is not None:
            for rel in relationships:
                count = struct.unpack("<I", in_stream.read(4))[0]
                ids = []
                for _ in range(0, count):
                    ids.append(struct.unpack("<Q", in_stream.read(8))[0])
                result.relations[rel.id] = ids

        return result


@dataclass
class PointAnnotation(Annotation):
    """Point annotation represented by a single 3D position."""

    position: Sequence[float] = field(default_factory=lambda: (0.0, 0.0, 0.0))

    GEOMETRY_BYTES: ClassVar[int] = 12  # position (3 floats)

    # We define an explicit initializer in order to control the parameter order;
    # id and points are required, properties and relations are optional.
    def __init__(
        self,
        id: int | None = None,  # pylint: disable=redefined-builtin
        position: Sequence[float] = (0.0, 0.0, 0.0),
        properties: dict[str, Any] | None = None,
        relations: dict[str, int | list[int]] | None = None,
    ):
        if id is None:
            id = random.randint(0, 2 ** 64 - 1)
        if properties is None:
            properties = {}
        if relations is None:
            relations = {}

        super().__init__(id=id, properties=properties, relations=relations)
        self.position = position

    def write_geometry(self, output: IO[bytes]) -> None:
        """Write position as float32le values."""
        output.write(struct.pack("<3f", *self.position))

    def get_geometry_byte_size(self) -> int:
        """Return the number of bytes used by geometry data."""
        return self.GEOMETRY_BYTES

    @classmethod
    def read_geometry(cls, in_stream: IO[bytes]) -> "PointAnnotation":
        """Read point geometry from binary format."""
        position = struct.unpack("<3f", in_stream.read(12))
        return cls(position=position)

    def in_bounds(
        self, bounds, resolution: Sequence[float] | None = None, strict: bool = False
    ) -> bool:
        """Return whether this point is in the given bounds.

        :param bounds: BBox3D or VolumetricIndex.
        :param point_resolution: Resolution of this point; defaults to [1,1,1].
        """
        if resolution is None:
            resolution = [1, 1, 1]
        if hasattr(bounds, "resolution"):
            to_res = bounds.resolution
            from_res = Vec3D(*resolution)
            position = tuple(round(Vec3D(*self.position) * from_res / to_res, VEC3D_PRECISION))
            return bounds.contains(position)
        else:
            return bounds.contains(self.position, resolution)

    def convert_coordinates(self, from_res, to_res) -> None:
        """Convert coordinates from one resolution to another (mutates instance).

        :param from_res: Source resolution (Vec3D)
        :param to_res: Target resolution (Vec3D)
        """
        self.position = tuple(round(Vec3D(*self.position) * from_res / to_res, VEC3D_PRECISION))

    def with_converted_coordinates(self, from_res, to_res) -> "PointAnnotation":
        """Return new PointAnnotation with converted coordinates (immutable).

        :param from_res: Source resolution (Vec3D)
        :param to_res: Target resolution (Vec3D)
        """
        new_position = tuple(round(Vec3D(*self.position) * from_res / to_res, VEC3D_PRECISION))
        return PointAnnotation(
            id=self.id, position=new_position, properties=self.properties, relations=self.relations
        )


@dataclass(init=False)
class LineAnnotation(Annotation):
    """Line annotation represented by two endpoint positions."""

    start: Sequence[float] = (0.0, 0.0, 0.0)
    end: Sequence[float] = (0.0, 0.0, 0.0)

    GEOMETRY_BYTES: ClassVar[int] = 24  # start (3 floats) + end (3 floats)

    # We define an explicit initializer in order to control the parameter order:
    # start and end are required; id, properties, and relations are optional.
    def __init__(
        self,
        start: Sequence[float] = (0.0, 0.0, 0.0),
        end: Sequence[float] = (0.0, 0.0, 0.0),
        id: int | None = None,  # pylint: disable=redefined-builtin
        properties: dict[str, Any] | None = None,
        relations: dict[str, int | list[int]] | None = None,
    ):
        if id is None:
            id = random.randint(0, 2 ** 64 - 1)
        if properties is None:
            properties = {}
        if relations is None:
            relations = {}

        super().__init__(id=id, properties=properties, relations=relations)
        self.start = start
        self.end = end

    def write_geometry(self, output: IO[bytes]) -> None:
        """Write start and end positions as float32le values."""
        output.write(struct.pack("<3f", *self.start))
        output.write(struct.pack("<3f", *self.end))

    def get_geometry_byte_size(self) -> int:
        """Return the number of bytes used by geometry data."""
        return self.GEOMETRY_BYTES

    @classmethod
    def read_geometry(cls, in_stream: IO[bytes]) -> "LineAnnotation":
        """Read line geometry from binary format."""
        start = struct.unpack("<3f", in_stream.read(12))
        end = struct.unpack("<3f", in_stream.read(12))
        return cls(start=start, end=end)

    def in_bounds(
        self, bounds, resolution: Sequence[float] | None = None, strict: bool = False
    ) -> bool:
        """Return whether this line anywhere intersects the given bounds.

        :param bounds: VolumetricIndex or BBox3D
        :param line_resolution: Resolution of the start and end points of this line.
        """
        if resolution is None:
            resolution = [1, 1, 1]
        if hasattr(bounds, "resolution"):
            to_res = bounds.resolution
            from_res = Vec3D(*resolution)
            start = tuple(round(Vec3D(*self.start) * from_res / to_res, VEC3D_PRECISION))
            end = tuple(round(Vec3D(*self.end) * from_res / to_res, VEC3D_PRECISION))
        else:
            start = tuple(self.start)
            end = tuple(self.end)
        if strict:
            return bounds.contains(start) and bounds.contains(end)
        else:
            if isinstance(bounds, BBox3D):
                return bounds.line_intersects(start, end, resolution=(1, 1, 1))
            else:
                return bounds.line_intersects(start, end)

    def convert_coordinates(self, from_res, to_res) -> None:
        """Convert coordinates from one resolution to another (mutates instance).

        :param from_res: Source resolution (Vec3D)
        :param to_res: Target resolution (Vec3D)
        """
        self.start = tuple(round(Vec3D(*self.start) * from_res / to_res, VEC3D_PRECISION))
        self.end = tuple(round(Vec3D(*self.end) * from_res / to_res, VEC3D_PRECISION))

    def with_converted_coordinates(self, from_res, to_res) -> "LineAnnotation":
        """Return new LineAnnotation with converted coordinates (immutable).

        :param from_res: Source resolution (Vec3D)
        :param to_res: Target resolution (Vec3D)
        """
        new_start = tuple(round(Vec3D(*self.start) * from_res / to_res, VEC3D_PRECISION))
        new_end = tuple(round(Vec3D(*self.end) * from_res / to_res, VEC3D_PRECISION))
        return LineAnnotation(
            id=self.id,
            start=new_start,
            end=new_end,
            properties=self.properties,
            relations=self.relations,
        )


@dataclass
class SpatialEntry:
    """
    This is a helper class, mainly used internally, to define each level of subdivision
    in a spatial (precomputed annotation) file.  A level is defined by:

    chunk_size: 3-element list or tuple defining the size of a chunk in X, Y, and Z (in voxels).
    grid_shape: 3-element list/tuple defining how many chunks there are in each dimension.
    key: a string e.g. "spatial1" used as a prefix for the chunk file on disk.
    limit: affects how the data is subsampled for display; should generally be the max number
      of annotations in any chunk at this level, or 1 for no subsampling.  It's confusing, but
      see:
      https://github.com/google/neuroglancer/issues/227#issuecomment-2246350747
    """

    chunk_size: Sequence[int]
    grid_shape: Sequence[int]
    key: str
    limit: int
    sharding: ShardingSpec | None = None

    def __post_init__(self):
        # Convert sequences to tuples to maintain the original behavior
        self.chunk_size = tuple(self.chunk_size)
        self.grid_shape = tuple(self.grid_shape)

    def to_json(self):
        result = f"""{{
            "chunk_size" : {list(self.chunk_size)},
            "grid_shape" : {list(self.grid_shape)},
            "key" : "{self.key}",
            "limit": {self.limit}"""
        if self.sharding is not None:
            result += ',\n            "sharding": ' + self.sharding.to_json()
        return result + "\n}"


def validate_spatial_entries(spatial_entries):
    """
    Validate a sequence of SpatialEntry objects for multi-level spatial indexing.

    :param spatial_entries: Sequence of SpatialEntry objects to validate
    :type spatial_entries: Sequence[SpatialEntry]
    :raises ValueError: If validation fails
    :returns: True if all validation checks pass
    :rtype: bool
    """
    if len(spatial_entries) < 2:
        return True  # Single or empty entries are trivially valid

    for i in range(1, len(spatial_entries)):
        prev_entry = spatial_entries[i - 1]
        curr_entry = spatial_entries[i]
        dimensions = range(len(curr_entry.chunk_size))

        # Check condition 1: chunk sizes should be equal or smaller in each dimension
        for dim in dimensions:
            if curr_entry.chunk_size[dim] > prev_entry.chunk_size[dim]:
                raise ValueError(
                    f"Entry {i}: chunk_size[{dim}] ({curr_entry.chunk_size[dim]}) "
                    f"is larger than previous entry's chunk_size[{dim}] "
                    f"({prev_entry.chunk_size[dim]})"
                )

        # Check condition 2: grid_shape * chunk_size should equal previous grid_shape * chunk_size
        for dim in dimensions:
            prev_total = prev_entry.grid_shape[dim] * prev_entry.chunk_size[dim]
            curr_total = curr_entry.grid_shape[dim] * curr_entry.chunk_size[dim]

            if prev_total != curr_total:
                raise ValueError(
                    f"Entry {i}: total size in dimension {dim} "
                    f"({curr_entry.grid_shape[dim]} * "
                    f"{curr_entry.chunk_size[dim]} = {curr_total}) "
                    f"does not match previous entry's total size "
                    f"({prev_entry.grid_shape[dim]} * "
                    f"{prev_entry.chunk_size[dim]} = {prev_total})"
                )

    return True


def get_child_cell_ranges(
    spatial_specs: Sequence[SpatialEntry], parent_level: int, parent_cell_index: tuple[int, ...]
) -> tuple[tuple[int, int], ...]:
    """
    Calculate the range of child cell indices at level parent_level+1 that fall within
    the given parent cell at parent_level.

    :param spatial_specs: ordered list of SpatialEntry defining subdivision levels
    :param parent_level: The level of the parent cell
    :param parent_cell_index: The cell index of the parent cell
    :returns: Tuple of (start, end) ranges for each dimension
    """
    parent_spec = spatial_specs[parent_level]
    child_spec = spatial_specs[parent_level + 1]

    ranges = []
    dimensions = range(len(parent_cell_index))
    for dim in dimensions:
        # Calculate how many child cells fit in one parent cell for this dimension
        cells_per_parent = child_spec.grid_shape[dim] // parent_spec.grid_shape[dim]

        # Calculate the start and end indices for child cells
        start_idx = parent_cell_index[dim] * cells_per_parent
        end_idx = start_idx + cells_per_parent

        ranges.append((start_idx, end_idx))

    return tuple(ranges)
