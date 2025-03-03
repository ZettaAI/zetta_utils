"""
Module to support writing of annotations in precomputed format.

(Note: unlike the other files in this directory, this one has
nothing to do with storage of annotations in a database.  It's
all about writing -- and to some extent, reading -- precomputed
files as used by Neuroglancer.)

References:
	https://github.com/google/neuroglancer/issues/227
	https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/annotations.md
"""

import io
import itertools
import json
import os
import struct
from math import ceil
from random import shuffle
from typing import IO, Any, Literal, Optional, Sequence, Tuple

import attrs
from cloudfiles import CloudFile, CloudFiles

from zetta_utils import builder, log, mazepa
from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.geometry.vec import VEC3D_PRECISION
from zetta_utils.layer.volumetric.index import VolumetricIndex

logger = log.get_logger("zetta_utils")


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


def to_3_tuple(value: Any) -> Tuple[float, ...]:
    # Ensure the value is a 3-element tuple of floats
    if len(value) != 3:
        raise ValueError("3-element sequence expected")
    return tuple(float(x) for x in value)


@attrs.define
class LineAnnotation:
    """
    LineAnnotation represents a Neuroglancer line annotation.  Start and end
    points are in voxels -- i.e., the coordinates are in units defined by
    "dimensions" in the info file, or some other resolution specified by the
    user.  (Like a Vec3D, context is needed to interpret these coordinates.)
    """

    BYTES_PER_ENTRY = 24  # start (3 floats), end (3 floats)

    id: int
    start: Tuple[float, ...] = attrs.field(converter=to_3_tuple)
    end: Tuple[float, ...] = attrs.field(converter=to_3_tuple)

    def write(self, output: IO[bytes]):
        """
        Write this annotation in binary format to the given output writer.
        """
        output.write(struct.pack("<3f", *self.start))
        output.write(struct.pack("<3f", *self.end))
        # NOTE: if you change or add to the above, be sure to also
        # change BYTES_PER_ENTRY accordingly.

    @staticmethod
    def read(in_stream: IO[bytes]):
        """
        Read an annotation in binary format from the given input reader.
        """
        return LineAnnotation(
            0,  # (ID will be filled in later)
            struct.unpack("<3f", in_stream.read(12)),
            struct.unpack("<3f", in_stream.read(12)),
        )

    def in_bounds(self, bounds: VolumetricIndex, strict=False):
        """
        Check whether this line at all crosses (when strict=False), or
        is entirely within (when strict=True), the given VolumetricIndex.
        (Assumes our coordinates match that of the index.)
        """
        if strict:
            return bounds.contains(self.start) and bounds.contains(self.end)
        return bounds.line_intersects(self.start, self.end)

    def convert_coordinates(self, from_res: Vec3D, to_res: Vec3D):
        """
        Convert our start and end coordinates from one resolution to another.
        Mutates the current instance.
        """
        self.start = tuple(round(Vec3D(*self.start) * from_res / to_res, VEC3D_PRECISION))
        self.end = tuple(round(Vec3D(*self.end) * from_res / to_res, VEC3D_PRECISION))

    def with_converted_coordinates(self, from_res: Vec3D, to_res: Vec3D):
        """
        Return a new LineAnnotation instance with converted coordinates.
        Does not mutate the current instance.
        """
        new_start = tuple(round(Vec3D(*self.start) * from_res / to_res, VEC3D_PRECISION))
        new_end = tuple(round(Vec3D(*self.end) * from_res / to_res, VEC3D_PRECISION))
        return LineAnnotation(self.id, new_start, new_end)


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

    def __init__(self, chunk_size: Sequence[int], grid_shape: Sequence[int], key: str, limit: int):
        self.chunk_size = chunk_size
        self.grid_shape = grid_shape
        self.key = key
        self.limit = limit

    def __repr__(self):
        return (
            f"SpatialEntry(chunk_size={self.chunk_size}, grid_shape={self.grid_shape}, "
            f'key="{self.key}", limit={self.limit})'
        )

    def to_json(self):
        return f"""{{
            "chunk_size" : {list(self.chunk_size)},
            "grid_shape" : {list(self.grid_shape)},
            "key" : "{self.key}",
            "limit": {self.limit}
        }}"""


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


def write_lines(file_or_gs_path: str, lines: Sequence[LineAnnotation], randomize: bool = True):
    """
    Write a set of lines to the given file, in 'multiple annotation encoding' format:
            1. Line count (uint64le)
            2. Data for each line (excluding ID), one after the other
            3. The line IDs (also as uint64le)

    :param file_path: local file or GS path of file to write
    :param lines: iterable of LineAnnotation objects
    :param randomize: if True, the lines will be written in random
            order (without mutating the lines parameter)
    """
    lines = list(lines)
    if randomize:
        lines = lines[:]
        shuffle(lines)

    buffer = io.BytesIO()
    # first write the count
    buffer.write(struct.pack("<Q", len(lines)))

    # then write the line data
    for line in lines:
        line.write(buffer)

    # finally write the ids at the end of the buffer
    for line in lines:
        buffer.write(struct.pack("<Q", line.id))

    buffer.seek(0)  # Rewind buffer to the beginning
    write_bytes(file_or_gs_path, buffer.getvalue())


def line_count_from_file_size(file_size: int) -> int:
    """
    Provide a count (or at least a very good estimate) of the number of lines in
    a line chunk file of the given size in bytes.
    """
    return round((file_size - 8) / (LineAnnotation.BYTES_PER_ENTRY + 8))


def count_lines_in_file(file_or_gs_path: str) -> int:
    """
    Provide a count (or at least a very good estimate) of the number of lines in
    the given line chunk file, as quickly as possible.
    """
    # We could open the file and read the count in the first 8 bytes.
    # But even faster is to just calculate it from the file length.
    cf = CloudFile(file_or_gs_path)
    return line_count_from_file_size(cf.size() or 0)


def read_bytes(file_or_gs_path: str):
    """
    Read bytes from a local file or Google Cloud Storage.

    :param file_or_gs_path: path to file to read (local or GCS path)
    :return: bytes read from the file
    """
    if "//" not in file_or_gs_path:
        file_or_gs_path = "file://" + file_or_gs_path
    cf = CloudFile(file_or_gs_path)
    return cf.get()


def read_lines(file_or_gs_path: str) -> list[LineAnnotation]:
    """
    Read a set of lines from the given file, which should be in
    'multiple annotation encoding' as defined in write_lines above.
    """
    data = read_bytes(file_or_gs_path)
    lines: list[LineAnnotation] = []
    if data is None or len(data) == 0:
        return lines
    with io.BytesIO(data) as buffer:
        # first read the count
        line_count = struct.unpack("<Q", buffer.read(8))[0]

        # then read the line data
        for _ in range(line_count):
            line = LineAnnotation.read(buffer)
            lines.append(line)

        # finally read the ids at the end of the buffer
        for i in range(line_count):
            line_id = struct.unpack("<Q", buffer.read(8))[0]
            lines[i].id = line_id
    return lines


def format_info(dimensions, lower_bound, upper_bound, spatial_data):
    spatial_json = "    " + ",\n        ".join([se.to_json() for se in spatial_data])
    return f"""{{
    "@type" : "neuroglancer_annotations_v1",
    "annotation_type" : "LINE",
    "by_id" : {{ "key" : "by_id" }},
    "dimensions" : {str(dimensions).replace("'", '"')},
    "lower_bound" : {list(lower_bound)},
    "properties" : [],
    "relationships" : [],
    "spatial" : [
    {spatial_json}
    ],
    "upper_bound" : {list(upper_bound)}
}}
"""


def write_info(dir_path, dimensions, lower_bound, upper_bound, spatial_data):
    """
    Write out the info (JSON) file describing a precomputed annotation file
    into the given directory.

    :param dir_path: path to file to write (local or GCS path)
    :param dimensions: dict providing x, y, and z dimensions e.g. [5, "nm"]
    :lower_bound: start of the data volume (in voxels)
    :upper_bound: end of the data volume (in voxels)
    :spatial_data: list of SpatialEntry objects
    """
    file_path = path_join(dir_path, "info")  # (note: not info.json as you would expect)
    info_content = format_info(dimensions, lower_bound, upper_bound, spatial_data)
    write_bytes(file_path, info_content.encode("utf-8"))


def parse_info(info_json):
    """
    Parse the given info file (in JSON format), and return:

     dimensions (dict), lower_bound, upper_bound, spatial_data (tuple of SpatialEntry)
    """
    data = json.loads(info_json)

    dimensions = data["dimensions"]
    lower_bound = data["lower_bound"]
    upper_bound = data["upper_bound"]
    spatial_data = tuple(
        SpatialEntry(entry["chunk_size"], entry["grid_shape"], entry["key"], entry["limit"])
        for entry in data["spatial"]
    )

    return dimensions, lower_bound, upper_bound, spatial_data


def read_info(dir_path):
    """
    Read the info file within the given directory, and return:

     dimensions (dict), lower_bound, upper_bound, spatial_data (tuple of SpatialEntry)

    If the file is empty or does not exist, return (None, None, None, None)
    """
    file_path = path_join(dir_path, "info")  # (note: not info.json as you would expect)
    try:
        data = read_bytes(file_path)
    except NotADirectoryError:
        data = None
    if data is None or len(data) == 0:
        return (None, None, None, None)
    return parse_info(data.decode("utf-8"))


def read_data(dir_path, spatial_entry):
    """
    Read all the line annotations in the given precomputed file hierarchy
    which are under the given spatial entry.  Normally this would be the
    finest spatial entry (smallest chunk_size, biggest grid_shape), as that's
    the only one guaranteed to contain all the data.  But it's up to the
    caller; if you really want to read from some other chunk size, feel free.
    """
    se = spatial_entry
    result = []
    for x in range(0, se.grid_shape[0]):
        for y in range(0, se.grid_shape[1]):
            for z in range(0, se.grid_shape[2]):
                level_dir = path_join(dir_path, se.key)
                anno_file_path = path_join(level_dir, f"{x}_{y}_{z}")
                result += read_lines(anno_file_path)
    return result


# pylint: disable=too-many-locals,too-many-nested-blocks,cell-var-from-loop
def subdivide(data, bounds: VolumetricIndex, chunk_sizes, write_to_dir=None, levels_to_write=None):
    """
    Subdivide the given data and bounds into chunks and subchunks of
    arbitrary depth, per the given chunk_sizes.  Return a list of
    SpatialEntry objects suitable for creating the info file.
    If write_to_dir is not None, then also write out the binary
    files ('multiple annotation encoding') for each chunk under
    subdirectories named with the appropriate keys, for all levels
    specified (by number) in levels_to_write (defaults to all).
    """
    if levels_to_write is None:
        levels_to_write = range(0, len(chunk_sizes))
    spatial_entries = []
    bounds_size = bounds.shape
    for level, chunk_size_seq in enumerate(chunk_sizes):
        chunk_size: Vec3D = Vec3D(*chunk_size_seq)
        limit = 0
        grid_shape = ceil(bounds_size / chunk_size)
        logger.info(f"subdividing {bounds} by {chunk_size}, for grid_shape {grid_shape}")
        level_key = f"spatial{level}"
        # total_qty = grid_shape[0] * grid_shape[1] * grid_shape[2]
        qty_done = 0
        for x in range(grid_shape[0]):
            x_start = bounds.start[0] + x * chunk_size[0]
            x_end = x_start + chunk_size[0]
            x_idx = VolumetricIndex.from_coords(
                (x_start, bounds.start[1], bounds.start[2]),
                (x_end, bounds.stop[1], bounds.stop[2]),
                bounds.resolution,
            )
            data_within_x = list(filter(lambda d: d.in_bounds(x_idx), data))
            for y in range(grid_shape[1]):
                y_start = bounds.start[1] + y * chunk_size[1]
                y_end = y_start + chunk_size[1]
                y_idx = VolumetricIndex.from_coords(
                    (x_start, y_start, bounds.start[2]),
                    (x_end, y_end, bounds.stop[2]),
                    bounds.resolution,
                )
                data_within_xy = list(filter(lambda d: d.in_bounds(y_idx), data_within_x))
                for z in range(grid_shape[2]):
                    qty_done += 1
                    chunk_start = bounds.start + Vec3D(x, y, z) * chunk_size
                    chunk_end = chunk_start + chunk_size
                    chunk_bounds = VolumetricIndex.from_coords(
                        chunk_start, chunk_end, bounds.resolution
                    )
                    # pylint: disable=cell-var-from-loop
                    chunk_data: Sequence[LineAnnotation] = list(
                        filter(lambda d: d.in_bounds(chunk_bounds), data_within_xy)
                    )
                    # logger.info(f'spatial{level}/{x}_{y}_{z} contains {len(chunk_data)} lines')
                    limit = max(limit, len(chunk_data))
                    if write_to_dir is not None and level in levels_to_write:
                        level_dir = path_join(write_to_dir, level_key)
                        if not os.path.exists(level_dir):
                            os.makedirs(level_dir)
                        anno_file_path = path_join(level_dir, f"{x}_{y}_{z}")
                        write_lines(anno_file_path, chunk_data)
        spatial_entries.append(SpatialEntry(chunk_size, grid_shape, level_key, limit))

    return spatial_entries


@builder.register("AnnotationLayer")
@attrs.define(init=False)
class AnnotationLayer:
    """
    This class represents a spatial (precomputed annotation) file.  It knows its data
    bounds, and how that is broken up into chunks (possibly over several levels).
    Methods are provided to write and update this data on disk.

    Note that on disk, this is actually a hierarchy of files.  At the root level is
    an "info" file (a JSON file, despite lacking a ".json" extension), and then a
    subdirectory for each subdivision level ("spatial0", "spatial1", etc.).  Within
    each subdirectory, there is a binary file for each chunk, named by its position
    within the grid, e.g. "1_2_0".  This class manages all that so you shouldn't
    have to worry about it.
    """

    index: VolumetricIndex
    chunk_sizes: Sequence[Sequence[int]]
    path: str = ""

    def __init__(
        self,
        path: str,
        index: Optional[VolumetricIndex] = None,
        chunk_sizes: Optional[Sequence[Sequence[int]]] = None,
    ):
        """
        Initialize an AnnotationLayer.

        :param path: local file or GS path of root of file hierarchy
        :param index: bounding box and resolution defining volume containing the data
        :param chunk_sizes: list of 3-element tuples/lists defining chunk sizes,
            in voxels (defaults to a single chunk containing the entire bounds)

        Note that index may be omitted ONLY if this is an existing file, in which case
        it will be inferred from the info file on disk.  But chunk_sizes may be omitted
        even for new files; in this case, the chunk size will be set to the full bounds
        (so you get only one spatial level).
        """
        assert path, "path parameter is required"
        if index is None:
            dims, lower_bound, upper_bound, spatial_entries = read_info(path)
            if dims is None:
                raise ValueError("index is required when file does not exist")  # pragma: no cover
            resolution = []
            for i in [0, 1, 2]:
                numAndUnit = dims["xyz"[i]]
                assert numAndUnit[1] == "nm", "Only dimensions in 'nm' are supported for reading"
                resolution.append(numAndUnit[0])
            # pylint: disable=E1120
            index = VolumetricIndex.from_coords(lower_bound, upper_bound, Vec3D(*resolution))
            chunk_sizes = [se.chunk_size for se in spatial_entries]
            logger.info(f"Inferred resolution: {resolution}")
            logger.info(f"Inferred chunk sizes: {chunk_sizes}")

        if chunk_sizes is None:
            chunk_sizes = [tuple(index.shape)]
        self.path = os.path.expanduser(path)
        self.index = index
        self.chunk_sizes = chunk_sizes

    def exists(self) -> bool:
        """
        Return whether this spatial file (more specifically, its info file) already exists.
        """
        path = path_join(self.path, "info")
        try:
            data = read_bytes(path)
        except NotADirectoryError:
            data = None
        return data is not None and len(data) > 0

    def delete(self):
        """
        Completely delete this precomputed annotation file.
        """
        # Delete all files under our path
        path = self.path
        if "//" not in path:
            path = "file://" + path
        cf = CloudFiles(path)
        cf.delete(cf.list())
        if path.startswith("file://"):
            # also delete the empty directories (which sadly CloudFiles cannot do)
            local_path = path[len("file://") :]
            for root, dirs, _ in os.walk(local_path, topdown=False):
                for directory in dirs:
                    os.rmdir(os.path.join(root, directory))

    def clear(self):
        """
        Clear out all data, and (re)write the info file, leaving an empty spatial
        file ready to pour fresh data into.
        """
        # Delete all files under our path
        self.delete()

        # Then, write (or overwrite) the info file
        self.write_info_file()

    def get_spatial_entries(self, limit_value: int = 1):
        """
        Subdivide our bounds according to our chunk_sizes, returning a list
        of SpatialEntry objects.  Use the given limit_value as the value of
        'limit' for each entry.  (1 is a good value since it results in no
        subsampling at any level in Neuroglancer.)
        """
        result = []
        bounds_size = self.index.shape
        for level, chunk_size_seq in enumerate(self.chunk_sizes):
            chunk_size = Vec3D(*chunk_size_seq)
            grid_shape = ceil(bounds_size / chunk_size)
            logger.info(f"subdividing {bounds_size} by {chunk_size}, for grid_shape {grid_shape}")
            level_key = f"spatial{level}"
            result.append(SpatialEntry(chunk_size, grid_shape, level_key, limit_value))
        return result

    def write_info_file(self, spatial_data: Optional[Sequence[SpatialEntry]] = None):
        """
        Write out just the info (JSON) file, with current parameters.
        """
        if spatial_data is None:
            spatial_data = self.get_spatial_entries()
        resolution = self.index.resolution
        write_info(
            dir_path=self.path,
            dimensions={
                "x": [resolution.x, "nm"],
                "y": [resolution.y, "nm"],
                "z": [resolution.z, "nm"],
            },
            lower_bound=self.index.start,
            upper_bound=self.index.stop,
            spatial_data=spatial_data,
        )

    def write_annotations(
        self,
        annotations: Sequence[LineAnnotation],
        annotation_resolution: Optional[Vec3D] = None,
        all_levels: bool = True,
        clearing_bbox: Optional[BBox3D] = None,
    ):  # pylint: disable=too-many-branches
        """
        Write a set of line annotations to the file, adding to any already there.

        :param annotations: sequence of LineAnnotations to add.
        :param annotation_resolution: resolution of given LineAnnotation coordinates;
        if not specified, assumes native coordinates (i.e. self.index.resolution)
        :param all_levels: if true, write to all spatial levels (chunk sizes).
            If false, write only to the lowest level (smallest chunks).
        :param clearing_bbox: if given, clear any existing data within these bounds;
            annotations must be entirely within these bounds.
        """
        if not annotations:
            logger.info("write_annotations called with 0 annotations to write")
            return
        if annotation_resolution and annotation_resolution != self.index.resolution:
            annotations = [
                x.with_converted_coordinates(annotation_resolution, self.index.resolution)
                for x in annotations
            ]
        qty_levels = len(self.chunk_sizes)
        levels = range(0, qty_levels) if all_levels else [qty_levels - 1]
        bounds_size = self.index.shape

        clearing_idx: Optional[VolumetricIndex] = None
        if clearing_bbox:
            clearing_idx = VolumetricIndex.from_coords(
                round(clearing_bbox.start / self.index.resolution),
                round(clearing_bbox.end / self.index.resolution),
                self.index.resolution,
            )
            if not all(map(lambda x: x.in_bounds(clearing_idx, strict=True), annotations)):
                raise ValueError("All annotations must be strictly within clearing_bbox")

        for level in levels:
            limit = 0
            chunk_size = Vec3D(*self.chunk_sizes[level])
            grid_shape = ceil(bounds_size / chunk_size)
            logger.info(f"subdividing {bounds_size} by {chunk_size}, for grid_shape {grid_shape}")
            level_key = f"spatial{level}"
            level_dir = path_join(self.path, level_key)
            if is_local_filesystem(self.path):
                os.makedirs(level_dir, exist_ok=True)

            # Yes, there are terser ways to do this 3D iteration in Python,
            # but they result in having to filter the full set of annotations
            # for every chunk, which turns out to be very slow.  Much faster
            # is to do one level at a time, filtering at each level, so that
            # the final filters don't have to wade through as much data.
            for x in range(grid_shape[0]):
                logger.debug(f"x = {x} of {grid_shape[0]}")
                split_by_x = VolumetricIndex.from_coords(
                    (
                        self.index.start[0] + x * chunk_size[0],
                        self.index.start[1],
                        self.index.start[2],
                    ),
                    (
                        self.index.start[0] + (x + 1) * chunk_size[0],
                        self.index.stop[1],
                        self.index.stop[2],
                    ),
                    self.index.resolution,
                )
                # pylint: disable=cell-var-from-loop
                split_data_by_x: list[LineAnnotation] = list(
                    filter(lambda d: d.in_bounds(split_by_x), annotations)
                )
                logger.debug(f":  {len(split_data_by_x)} lines")
                if not split_data_by_x:
                    continue
                for y in range(grid_shape[1]):
                    logger.debug(f"    y = {y} of {grid_shape[1]}")
                    split_by_y = VolumetricIndex.from_coords(
                        (
                            self.index.start[0] + x * chunk_size[0],
                            self.index.start[1] + y * chunk_size[1],
                            self.index.start[2],
                        ),
                        (
                            self.index.start[0] + (x + 1) * chunk_size[0],
                            self.index.start[1] + (y + 1) * chunk_size[1],
                            self.index.stop[2],
                        ),
                        self.index.resolution,
                    )
                    # pylint: disable=cell-var-from-loop
                    split_data_by_y: list[LineAnnotation] = list(
                        filter(lambda d: d.in_bounds(split_by_y), split_data_by_x)
                    )
                    logger.debug(f":  {len(split_data_by_y)} lines")
                    if not split_data_by_y:
                        continue
                    for z in range(grid_shape[2]):
                        split_by_z = VolumetricIndex.from_coords(
                            (
                                self.index.start[0] + x * chunk_size[0],
                                self.index.start[1] + y * chunk_size[1],
                                self.index.start[2] + z * chunk_size[2],
                            ),
                            (
                                self.index.start[0] + (x + 1) * chunk_size[0],
                                self.index.start[1] + (y + 1) * chunk_size[1],
                                self.index.start[2] + (z + 1) * chunk_size[2],
                            ),
                            self.index.resolution,
                        )
                        # Sanity check: manually compute the single-chunk bounds.
                        # It should be equal to split_by_z.
                        chunk_start = self.index.start + Vec3D(x, y, z) * chunk_size
                        chunk_end = chunk_start + chunk_size
                        chunk_bounds = VolumetricIndex.from_coords(
                            chunk_start, chunk_end, self.index.resolution
                        )
                        assert chunk_bounds == split_by_z
                        # pylint: disable=cell-var-from-loop
                        chunk_data: list[LineAnnotation] = list(
                            filter(lambda d: d.in_bounds(chunk_bounds), split_data_by_y)
                        )
                        if not chunk_data:
                            continue
                        anno_file_path = path_join(level_dir, f"{x}_{y}_{z}")
                        old_data = read_lines(anno_file_path)
                        if clearing_idx:
                            old_data = list(
                                filter(
                                    lambda d: not d.in_bounds(clearing_idx, strict=True), old_data
                                )
                            )
                        chunk_data += old_data
                        limit = max(limit, len(chunk_data))
                        write_lines(anno_file_path, chunk_data)

    def read_all(
        self,
        spatial_level: int = -1,
        filter_duplicates: bool = True,
        annotation_resolution: Optional[Vec3D] = None,
    ):
        """
        Read and return all annotations from the given spatial level.
        Note that an annotation that spans chunk boundaries will appear in
        multiple chunks.  In that case, the behavior of this function is
        determined by filter_duplicates: if filter_duplicates is True,
        then no annotation (by id) will appear in the results more than
        once, even if it spans chunk boundaries; but if it is False, then
        the same annotation may appear multiple times.
        """
        level = spatial_level if spatial_level >= 0 else len(self.chunk_sizes) + spatial_level
        result = []
        bounds_size = self.index.shape
        chunk_size = Vec3D(*self.chunk_sizes[level])
        grid_shape = ceil(bounds_size / chunk_size)
        level_key = f"spatial{level}"
        level_dir = path_join(self.path, level_key)
        # total_chunks = grid_shape[0] * grid_shape[1] * grid_shape[2]
        chunks_read = 0
        for x in range(0, grid_shape[0]):
            for y in range(0, grid_shape[1]):
                for z in range(0, grid_shape[2]):
                    chunks_read += 1
                    anno_file_path = path_join(level_dir, f"{x}_{y}_{z}")
                    result += read_lines(anno_file_path)
        if filter_duplicates:
            result_dict = {line.id: line for line in result}
            result = list(result_dict.values())
        if annotation_resolution:
            for line in result:
                line.convert_coordinates(self.index.resolution, annotation_resolution)
        return result

    def find_max_size(self, spatial_level: int = -1):
        """
        Find the maximum number of entries in any chunk at the given level.
        """
        level = spatial_level if spatial_level >= 0 else len(self.chunk_sizes) + spatial_level
        bounds_size = self.index.shape
        chunk_size = Vec3D(*self.chunk_sizes[level])
        grid_shape = ceil(bounds_size / chunk_size)
        level_key = f"spatial{level}"
        level_dir = path_join(self.path, level_key)
        if "//" not in level_dir:
            level_dir = "file://" + level_dir
        cf = CloudFiles(level_dir)
        file_paths = [
            f"{x}_{y}_{z}"
            for x, y, z in itertools.product(
                range(grid_shape[0]), range(grid_shape[1]), range(grid_shape[2])
            )
        ]
        file_sizes = cf.size(file_paths)
        max_file_size = max(x or 0 for x in file_sizes.values())
        return line_count_from_file_size(max_file_size)

    def read_in_bounds(
        self, roi: BBox3D, annotation_resolution: Optional[Vec3D] = None, strict: bool = False
    ):
        """
        Return all annotations within the given bounds (index).

        :param roi: region of interest
        :param annotation_resolution: resolution of returned LineAnnotation coordinates;
        if not specified, uses native coordinates (i.e. self.index.resolution)
        :param strict: if True, return ONLY annotations entirely within the given bounds;
        if False, then you may also get some annotations that are partially or entirely
        outside the given bounds
        :return: list of LineAnnotation objects
        """
        level = len(self.chunk_sizes) - 1
        result = []
        bounds_size_vx = self.index.shape
        chunk_size_vx = Vec3D(*self.chunk_sizes[level])
        grid_shape = ceil(bounds_size_vx / chunk_size_vx)
        level_key = f"spatial{level}"
        level_dir = path_join(self.path, level_key)

        roi_start_vx = round(roi.start / self.index.resolution)
        roi_end_vx = round(roi.end / self.index.resolution)
        roi_index = VolumetricIndex.from_coords(roi_start_vx, roi_end_vx, self.index.resolution)

        start_chunk = (roi_index.start - self.index.start) // chunk_size_vx
        end_chunk = (roi_index.stop - self.index.start) // chunk_size_vx
        for x in range(max(0, start_chunk[0]), min(grid_shape[0], end_chunk[0] + 1)):
            for y in range(max(0, start_chunk[1]), min(grid_shape[1], end_chunk[1] + 1)):
                for z in range(max(0, start_chunk[2]), min(grid_shape[2], end_chunk[2] + 1)):
                    anno_file_path = path_join(level_dir, f"{x}_{y}_{z}")
                    result += read_lines(anno_file_path)
        if strict:
            result = list(
                filter(lambda x: roi_index.contains(x.start) and roi_index.contains(x.end), result)
            )
        result_dict = {line.id: line for line in result}
        result = list(result_dict.values())
        if annotation_resolution:
            for line in result:
                line.convert_coordinates(self.index.resolution, annotation_resolution)
        return result

    def post_process(self):
        """
        Read all our data from the lowest-level chunks on disk, then rewrite:
          1. The higher-level chunks, if any; and
          2. The info file, with correct limits for each level.
        This is useful after writing out a bunch of data with
          write_annotations(data, False), which writes to only the lowest-level chunks.
        """
        if len(self.chunk_sizes) == 1:
            # Special case: only one chunk size, no subdivision.
            # In this case, we can cheat considerably.
            # Just iterate over the spatial entry files, getting the line
            # count in each one, and keep track of the max.
            max_line_count = self.find_max_size(0)
            # print(f"Found max_line_count = {max_line_count}")
            spatial_entries = self.get_spatial_entries(max_line_count)
        else:
            # Multiple chunk sizes means we have to start at the lowest
            # level, and re-subdivide it at each higher level.

            # read data (from lowest level chunks)
            all_data = self.read_all()

            # subdivide as if writing data to all levels EXCEPT the last one
            levels_to_write = range(0, len(self.chunk_sizes) - 1)
            spatial_entries = subdivide(
                all_data, self.index, self.chunk_sizes, self.path, levels_to_write
            )

        # rewrite the info file, with the updated spatial entries
        self.write_info_file(spatial_entries)


@builder.register("build_annotation_layer")
def build_annotation_layer(  # pylint: disable=too-many-locals, too-many-branches
    path: str,
    resolution: Sequence[float] | None = None,
    dataset_size: Sequence[int] | None = None,
    voxel_offset: Sequence[int] | None = None,
    index: VolumetricIndex | None = None,
    chunk_sizes: Sequence[Sequence[int]] | None = None,
    mode: Literal["read", "write", "replace", "update"] = "write",
) -> AnnotationLayer:  # pragma: no cover # trivial conditional, delegation only
    """Build an AnnotationLayer (spatially indexed annotations in precomputed file format).

    :param path: Path to the precomputed file (directory).
    :param resolution: (x, y, z) size of one voxel, in nm.
    :dataset_size: Precomputed dataset size (in voxels) for all scales.
    :param voxel_offset: start of the dataset volume (in voxels) for all scales.
    :index: VolumetricIndex indicating dataset size and resolution.  Note that
      for new files, you must provide either (resolution, dataset_size, voxel_offset)
      or index, but not both.  For existing files, all these are optional.
    :chunk_sizes: Chunk sizes for spatial index; defaults to a single chunk for
      new files (or the existing chunk structure for existing files).
    :mode: How the file should be created or opened:
       "read": for reading only; throws error if file does not exist.
       "write": for writing; throws error if file exists.
       "replace": for writing; if file exists, it is cleared of all data.
       "update": for writing additional data; throws error if file does not exist.
    """
    dims, lower_bound, upper_bound, spatial_entries = read_info(path)
    file_exists = spatial_entries is not None
    file_resolution = []
    file_index = None
    file_chunk_sizes = []
    if file_exists:
        for i in [0, 1, 2]:
            numAndUnit = dims["xyz"[i]]
            assert numAndUnit[1] == "nm", "Only dimensions in 'nm' are supported for reading"
            file_resolution.append(numAndUnit[0])
        # pylint: disable=E1120
        file_index = VolumetricIndex.from_coords(lower_bound, upper_bound, Vec3D(*file_resolution))
        file_chunk_sizes = [se.chunk_size for se in spatial_entries]

    if mode in ("read", "update") and not file_exists:
        raise IOError(
            f"AnnotationLayer built with mode {mode}, but file does not exist (path: {path})"
        )
    if mode == "write" and file_exists:
        raise IOError(
            f"AnnotationLayer built with mode {mode}, but file already exists (path: {path})"
        )

    if index is None:
        if mode == "write" or (mode == "replace" and not file_exists):
            if resolution is None:
                raise ValueError("when `index` is not provided, `resolution` is required")
            if dataset_size is None:
                raise ValueError("when `index` is not provided, `dataset_size` is required")
            if voxel_offset is None:
                raise ValueError("when `index` is not provided, `voxel_offset` is required")
            if len(resolution) != 3:
                raise ValueError(f"`resolution` needs 3 elements, not {len(resolution)}")
            if len(dataset_size) != 3:
                raise ValueError(f"`dataset_size` needs 3 elements, not {len(dataset_size)}")
            if len(voxel_offset) != 3:
                raise ValueError(f"`dataset_size` needs 3 elements, not {len(voxel_offset)}")
            end_coord = tuple(a + b for a, b in zip(voxel_offset, dataset_size))
            index = VolumetricIndex.from_coords(voxel_offset, end_coord, resolution)
    else:
        if resolution is not None:
            raise ValueError("providing both `index` and `resolution` is invalid")
        if dataset_size is not None:
            raise ValueError("providing both `index` and `dataset_size` is invalid")
        if voxel_offset is not None:
            raise ValueError("providing both `index` and `voxel_offset` is invalid")

    if mode == "update":
        if index is not None and index != file_index:
            raise ValueError(
                f'opened file for "update" with index {index}, '
                f"but existing file index is {file_index}"
            )
        if chunk_sizes is not None and chunk_sizes != file_chunk_sizes:
            raise ValueError(
                f'opened file for "update" with chunk_sizes {chunk_sizes}, '
                f"but existing file chunk_sizes is {file_chunk_sizes}"
            )

    sf = AnnotationLayer(path, index, chunk_sizes)
    if mode in ("write", "replace"):
        sf.clear()
    return sf


@mazepa.taskable_operation
def post_process_annotation_layer_op(target: AnnotationLayer):  # pragma: no cover
    target.post_process()


@builder.register("post_process_annotation_layer_flow")
@mazepa.flow_schema
def post_process_annotation_layer_flow(target: AnnotationLayer):  # pragma: no cover
    yield post_process_annotation_layer_op.make_task(target)
