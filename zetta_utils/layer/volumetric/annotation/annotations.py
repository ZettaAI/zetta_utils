"""
Data classes and support code for Neuroglancer annotations (primarily
in precomputed format).  Reference:

https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/annotations.md
"""

import io
import json
import os
import random
import re
import string
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from random import shuffle
from typing import IO, Any, ClassVar, Dict, List, Optional, Sequence, Union

from cloudfiles import CloudFile

from zetta_utils.geometry import Vec3D
from zetta_utils.geometry.vec import VEC3D_PRECISION


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
    description: Optional[str] = None
    enum_values: Optional[List[Union[int, float]]] = None
    enum_labels: Optional[List[str]] = None

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
            )

    def _validate_type(self):
        """Validate that the type is one of the supported types."""
        if self.type not in self.VALID_TYPES:
            raise ValueError(
                f"Property type '{self.type}' must be one of: "
                + ", ".join(sorted(self.VALID_TYPES))
            )

    def _validate_enums(self):
        """Validate enum_values and enum_labels consistency."""
        has_values = self.enum_values is not None
        has_labels = self.enum_labels is not None

        # Both must be provided together or not at all
        if has_values != has_labels:
            raise ValueError("enum_values and enum_labels must both be provided or both be None")

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
                )

    def to_dict(self) -> dict:
        """Convert to dictionary format matching the JSON specification."""
        result: Dict[str, Any] = {"id": self.id, "type": self.type}

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

    def to_json(self, indent: Optional[int] = None) -> str:
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
    key: Optional[str] = None

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
        return {"id": self.id, "key": self.key}

    @classmethod
    def from_dict(cls, data: dict) -> "Relationship":
        """Create Relationship from dictionary."""
        return cls(id=data["id"], key=data.get("key"))  # Will auto-generate if None

    def to_json(self, indent: Optional[int] = None) -> str:
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
    properties: Dict[str, Any] = field(default_factory=dict)
    relations: Dict[str, int | List[int]] = field(default_factory=dict)

    @abstractmethod
    def in_bounds(
        self, bounds, resolution: Optional[Sequence[float]] = None, strict: bool = False
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
        property_specs: Optional[Sequence[PropertySpec]] = None,
        relationships: Optional[Sequence[Relationship]] = None,
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
                if isinstance(related_ids, int):
                    related_ids = [related_ids]
                output.write(struct.pack("<I", len(related_ids)))
                for idnum in related_ids:
                    output.write(struct.pack("<Q", idnum))

    @classmethod
    def read(
        cls,
        in_stream: IO[bytes],
        type: str,  # pylint: disable=redefined-builtin
        property_specs: Optional[Sequence[PropertySpec]] = None,
        relationships: Optional[Sequence[Relationship]] = None,
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
        id: Optional[int] = None,  # pylint: disable=redefined-builtin
        position: Sequence[float] = (0.0, 0.0, 0.0),
        properties: Optional[Dict[str, Any]] = None,
        relations: Optional[Dict[str, Union[int, List[int]]]] = None,
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
        self, bounds, resolution: Optional[Sequence[float]] = None, strict: bool = False
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


@dataclass
class LineAnnotation(Annotation):
    """Line annotation represented by two endpoint positions."""

    start: Sequence[float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    end: Sequence[float] = field(default_factory=lambda: (0.0, 0.0, 0.0))

    GEOMETRY_BYTES: ClassVar[int] = 24  # start (3 floats) + end (3 floats)

    # We define an explicit initializer in order to control the parameter order;
    # id and points are required, properties and relations are optional.
    def __init__(
        self,
        id: Optional[int] = None,  # pylint: disable=redefined-builtin
        start: Sequence[float] = (0.0, 0.0, 0.0),
        end: Sequence[float] = (0.0, 0.0, 0.0),
        properties: Optional[Dict[str, Any]] = None,
        relations: Optional[Dict[str, Union[int, List[int]]]] = None,
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
        self, bounds, resolution: Optional[Sequence[float]] = None, strict: bool = False
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

    def __post_init__(self):
        # Convert sequences to tuples to maintain the original behavior
        self.chunk_size = tuple(self.chunk_size)
        self.grid_shape = tuple(self.grid_shape)

    def to_json(self):
        return f"""{{
            "chunk_size" : {list(self.chunk_size)},
            "grid_shape" : {list(self.grid_shape)},
            "key" : "{self.key}",
            "limit": {self.limit}
        }}"""


# pylint: disable=too-many-instance-attributes
class SimpleWriter:
    def __init__(self, anno_type, dimensions, lower_bound, upper_bound):
        """
        Initialize SimpleWriter with required parameters.

        :param anno_type: one of 'POINT', 'LINE' (and later others)
        :param dimensions: dimensions for the annotation space
        :param lower_bound: lower bound coordinates
        :param upper_bound: upper bound coordinates
        :param annotations: sequence of LineAnnotation objects (optional)
        """
        self.anno_type = anno_type
        self.dimensions = dimensions
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.annotations = []  # list of Annotation objects
        self.spatial_specs = []  # SpatialEntry objects
        self.property_specs = []  # PropertySpec objects
        self.relationships = []  # Relationship objects

    def format_info(self):
        """Format the info JSON structure using instance properties."""
        spatial_json = "    " + ",\n        ".join([se.to_json() for se in self.spatial_specs])
        property_json = "    " + ",\n        ".join([ps.to_json() for ps in self.property_specs])
        relationship_json = "    " + ",\n        ".join([r.to_json() for r in self.relationships])
        return f"""{{
        "@type" : "neuroglancer_annotations_v1",
        "annotation_type" : "{self.anno_type}",
        "by_id" : {{ "key" : "by_id" }},
        "dimensions" : {str(self.dimensions).replace("'", '"')},
        "lower_bound" : {list(self.lower_bound)},
        "properties" : [
        {property_json}
        ],
        "relationships" : [
        {relationship_json}
        ],
        "spatial" : [
        {spatial_json}
        ],
        "upper_bound" : {list(self.upper_bound)}
    }}
    """

    def write(self, dir_path: str):
        """
        Write all annotation data to the specified directory.

        :param dir_path: path to the directory where files will be written
        """
        # Write info file
        info_content = self.format_info()
        info_file_path = path_join(dir_path, "info")
        write_bytes(info_file_path, info_content.encode("utf-8"))

        # Write by-id index (including relationships)
        self._write_by_id_index(path_join(dir_path, "by_id"))

        # Write the spatial index
        self._write_spatial_index(dir_path)

        # Write the related-object-id indexes
        for rel in self.relationships:
            self._write_related_index(dir_path, rel)

    def _write_annotations(
        self,
        file_or_gs_path: str,
        annotations: Optional[Sequence[Annotation]] = None,
        randomize: bool = False,
    ):
        """
        Write a set of lines to the given file, in 'multiple annotation encoding' format:
                1. Line count (uint64le)
                2. Data for each line (excluding ID), one after the other
                3. The line IDs (also as uint64le)

        :param file_or_gs_path: local file or GS path of file to write
        :param annotations: iterable of Annotation objects (uses self.annotations if None)
        :param randomize: if True, the annotations will be written in random
                order (without mutating the lines parameter)
        """
        if annotations is None:
            annotations = self.annotations

        annotations = list(annotations)
        if randomize:
            annotations = annotations[:]
            shuffle(annotations)

        buffer = io.BytesIO()
        # first write the count
        buffer.write(struct.pack("<Q", len(annotations)))

        # then write the annotation data
        for anno in annotations:
            anno.write(buffer, self.property_specs)

        # finally write the ids at the end of the buffer
        for anno in annotations:
            buffer.write(struct.pack("<Q", anno.id))

        buffer.seek(0)  # Rewind buffer to the beginning
        write_bytes(file_or_gs_path, buffer.getvalue())

    def _write_by_id_index(self, by_id_path: str):
        """
        Write the Annotation id index for the given set of annotations.
        Currently, in unsharded uint64 index format.

        :param by_id_path: complete path to the by_id directory.
        """
        # In unsharded format, the by_id directory simply contains a little
        # binary file for each annotation, named with its id.
        for anno in self.annotations:
            file_path = path_join(by_id_path, str(anno.id))
            buffer = io.BytesIO()
            anno.write(buffer, self.property_specs, self.relationships)
            write_bytes(file_path, buffer.getvalue())

    def _write_spatial_index(self, dir_path: str):
        """
        Write the spatial index for the given set of annotations.  NOTE:
        this implementation is a quick hack that assumes only 1 spatial
        level, consisting of only 1 chunk (which contains all annotations).

        :param dir_path: path to the directory containing the info file
        """
        level = 0
        level_key = f"spatial{level}"
        level_dir = path_join(dir_path, level_key)
        anno_file_path = path_join(level_dir, "0_0_0")
        self._write_annotations(anno_file_path, self.annotations, True)

    def _write_related_index(self, dir_path: str, relation: Relationship):
        """
        Write a related object ID index, where for each related object ID,
        we have a file of annotations that contain that ID for that relation.

        :param dir_path: path to the directory containing the info file
        :param relation: the Relationship object to process
        """
        rel_id_to_anno: Dict[int, List[Annotation]] = {}
        for anno in self.annotations:
            related_ids = anno.relations.get(relation.id, [])
            if isinstance(related_ids, int):
                related_ids = [related_ids]
            for rel_id in related_ids:
                anno_list = rel_id_to_anno.get(rel_id, None)
                if anno_list is None:
                    anno_list = []
                    rel_id_to_anno[rel_id] = anno_list
                anno_list.append(anno)
        assert relation.key is not None  # which it can't be, silly black
        rel_dir_path = path_join(dir_path, relation.key)
        for related_id, anno_list in rel_id_to_anno.items():
            file_path = path_join(rel_dir_path, str(related_id))
            self._write_annotations(file_path, anno_list, False)


# pylint: enable=too-many-instance-attributes


def _line_demo(path):
    # Write out a simple line annotations file (with properties and relations)
    # to demonstrate usage.
    dimensions = {"x": [18, "nm"], "y": [18, "nm"], "z": [45, "nm"]}
    lower_bound = [53092, 56657, 349]
    upper_bound = [53730, 57135, 634]

    writer = SimpleWriter("LINE", dimensions, lower_bound, upper_bound)

    writer.spatial_specs.append(SpatialEntry([1024, 1024, 512], [1, 1, 1], "spatial0", 1))

    writer.property_specs.append(PropertySpec("score", "float32", "Score value in range [0,1]"))
    writer.property_specs.append(PropertySpec("score_pct", "uint8", "Int score in range [0,100]"))
    writer.property_specs.append(
        PropertySpec(
            "mood",
            "uint8",
            "Overall affect",
            [0, 1, 2, 3, 4],
            ["none", "sad", "neutral", "happy", "ecstatic"],
        )
    )

    writer.relationships.append(Relationship("Presyn Cell"))
    writer.relationships.append(Relationship("Postsyn Cell"))

    writer.annotations.append(
        LineAnnotation(
            id=1001,
            start=(53092, 56657, 349),
            end=(53730, 57135, 634),
            properties={"score": 0.95, "score_pct": 95, "mood": 1},
        )
    )
    writer.annotations.append(
        LineAnnotation(
            id=1002,
            start=(53400, 56900, 500),
            end=(53420, 56900, 500),
            properties={"score": 0.42, "score_pct": 42, "mood": 4},
        )
    )
    writer.annotations.append(
        LineAnnotation(
            id=1003,
            start=(53226, 56899, 460),
            end=(53265, 56899, 458),
            properties={"score": 0.5, "score_pct": 50, "mood": 2},
            relations={"Presyn Cell": 648518346453391624, "Postsyn Cell": 648518346439350172},
        )
    )
    writer.annotations.append(
        LineAnnotation(
            id=1004,
            start=(53127, 56899, 457),
            end=(53104, 56911, 457),
            properties={"score": 0.8, "score_pct": 80, "mood": 3},
            relations={"Presyn Cell": [648518346453391624], "Postsyn Cell": [648518346454006042]},
        )
    )

    writer.write(path)
    print(f"Wrote {path}")


def _point_demo(path):
    # Write out a simple point annotations file (with properties).
    dimensions = {"x": [18, "nm"], "y": [18, "nm"], "z": [45, "nm"]}
    lower_bound = [53092, 56657, 349]
    upper_bound = [53730, 57135, 634]

    writer = SimpleWriter("POINT", dimensions, lower_bound, upper_bound)

    writer.spatial_specs.append(SpatialEntry([1024, 1024, 512], [1, 1, 1], "spatial0", 1))

    writer.property_specs.append(PropertySpec("score", "float32", "Score value in range [0,1]"))
    writer.property_specs.append(PropertySpec("score_pct", "uint8", "Int score in range [0,100]"))
    writer.property_specs.append(
        PropertySpec(
            "mood",
            "uint8",
            "Overall affect",
            [0, 1, 2, 3, 4],
            ["none", "sad", "neutral", "happy", "ecstatic"],
        )
    )

    writer.annotations.append(
        PointAnnotation(
            id=1001,
            position=(53092, 56657, 349),
            properties={"score": 0.95, "score_pct": 95, "mood": 1},
        )
    )
    writer.annotations.append(
        PointAnnotation(
            id=1002,
            position=(53400, 56900, 500),
            properties={"score": 0.42, "score_pct": 42, "mood": 4},
        )
    )
    writer.annotations.append(
        PointAnnotation(
            id=1003,
            position=(53226, 56899, 460),
            properties={"score": 0.5, "score_pct": 50, "mood": 2},
        )
    )
    writer.annotations.append(
        PointAnnotation(
            id=1004,
            position=(53127, 56899, 457),
            properties={"score": 0.8, "score_pct": 80, "mood": 3},
        )
    )

    path = os.path.expanduser("~/temp/simple_anno_points")
    writer.write(path)
    print(f"Wrote {path}")


if __name__ == "__main__":
    _line_demo("~/temp/simple_anno_lines")
    _point_demo("~/temp/simple_anno_points")
