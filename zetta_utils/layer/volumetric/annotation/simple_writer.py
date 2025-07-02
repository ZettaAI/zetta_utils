"""
This module provides the SimpleWriter class, which is able to write a precomputed
annotation file given a complete set of annotations (including properties and
relations), including sharding support.
"""
# pylint: disable=too-many-instance-attributes,too-many-branches,too-many-nested-blocks,too-many-locals

import io
import os
import struct
from dataclasses import dataclass
from random import random, shuffle
from typing import Any, Dict, List, Optional, Sequence, Tuple

from zetta_utils.geometry import BBox3D
from zetta_utils.layer.volumetric.annotation.annotations import (
    Annotation,
    LineAnnotation,
    PointAnnotation,
    PropertySpec,
    Relationship,
    SpatialEntry,
    get_child_cell_ranges,
    validate_spatial_entries,
)
from zetta_utils.layer.volumetric.annotation.sharding import Chunk, write_shard_files
from zetta_utils.layer.volumetric.annotation.utilities import (
    compressed_morton_code,
    path_join,
    write_bytes,
)


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
        self.by_id_sharding = None  # optional ShardingSpec

    def format_info(self):
        """Format the info JSON structure using instance properties."""
        spatial_json = "    " + ",\n        ".join([se.to_json() for se in self.spatial_specs])
        property_json = "    " + ",\n        ".join([ps.to_json() for ps in self.property_specs])
        relationship_json = "    " + ",\n        ".join([r.to_json() for r in self.relationships])
        if self.by_id_sharding is None:
            by_id_json = '{ "key" : "by_id" }'
        else:
            by_id_json = '{ "key" : "by_id", "sharding": ' + self.by_id_sharding.to_json() + " }"

        return f"""{{
        "@type" : "neuroglancer_annotations_v1",
        "annotation_type" : "{self.anno_type}",
        "by_id" : {by_id_json},
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

    def compile_multi_annotation_buffer(
        self,
        annotations: Optional[Sequence[Annotation]] = None,
        randomize: bool = False,
    ):
        """
        Compile a set of lines to a bytes, in 'multiple annotation encoding' format:
                1. Line count (uint64le)
                2. Data for each line (excluding ID), one after the other
                3. The line IDs (also as uint64le)

        :param file_or_gs_path: local file or GS path of file to write
        :param annotations: iterable of Annotation objects (uses self.annotations if None)
        :param randomize: if True, the annotations will be written in random
                order (without mutating the lines parameter)
        :return: bytes representing compiled annotations
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
        return buffer.getvalue()

    def write_annotations(
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
        data = self.compile_multi_annotation_buffer(annotations, randomize)
        write_bytes(file_or_gs_path, data)

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
        self.write_annotations(anno_file_path, self.annotations, True)

    def _write_related_index(self, dir_path: str, relation: Relationship):
        """
        Write a related object ID index, where for each related object ID,
        we have a file of annotations that contain that ID for that relation.

        :param dir_path: path to the directory containing the info file
        :param relation: the Relationship object to process
        """
        # Gather up the annotations for each related value
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

        # Then write to disk directly, or prepare as shard files
        assert relation.key is not None  # which it can't be, silly black
        rel_dir_path = path_join(dir_path, relation.key)
        if relation.sharding is None:
            for related_id, anno_list in rel_id_to_anno.items():
                file_path = path_join(rel_dir_path, str(related_id))
                self.write_annotations(file_path, anno_list, False)
        else:
            chunks = []
            for related_id, anno_list in rel_id_to_anno.items():
                data = self.compile_multi_annotation_buffer(anno_list, False)
                chunks.append(Chunk(related_id, data))
                # print(f"Related id {related_id} compiles to {len(data)} bytes")
            write_shard_files(rel_dir_path, relation.sharding, chunks)

    def subdivide(self, dir_path: str, prob_per_level: Sequence[float]):
        """
        Subdivide annotations into a multi-level spatial index using a breadth-first approach.

        :param dir_path: Directory path for output files
        :param prob_per_level: probability (0-1) of emitting annotation at each level
        """

        @dataclass
        class SubdivideTask:
            """
            Represents a work task for processing annotations at a specific spatial level and cell.

            :param level: Spatial index level (0 = coarsest)
            :param cell_index: Cell coordinates as tuple (x, y, z indices within the grid)
            :param annotations: List of annotations to process for this cell
            """

            level: int
            cell_index: Tuple[int, ...]
            annotations: List[Any]

            def file_name(self):
                return "_".join(str(i) for i in self.cell_index)

        # Make sure our spatial entries make sense; and for now,
        # require that we have a single cell at the coarsest level
        validate_spatial_entries(self.spatial_specs)
        if self.spatial_specs[0].grid_shape != (1, 1, 1):
            raise ValueError("subdivide requires level 0 grid_shape to be (1,1,1)")
        if len(prob_per_level) != len(self.spatial_specs):
            raise ValueError(
                f"subdivide: prob_per_level needs {len(self.spatial_specs)}"
                " to match spatial levels"
            )

        # Make our task queue, with all annotations in the level 0 cell
        taskQ = []

        # Also prepare a list of chunks for each level (this will only be used for
        # levels that want sharding)
        chunks_per_level: List[List[Chunk]] = [[] for lvl in self.spatial_specs]

        # Push task for cell 0 - the single cell at the coarsest level containing all annotations
        initial_task = SubdivideTask(
            level=0, cell_index=(0, 0, 0), annotations=list(self.annotations)
        )
        taskQ.append(initial_task)

        max_level = len(self.spatial_specs) - 1

        # Process the task queue
        while taskQ:
            task = taskQ.pop(0)  # FIFO for breadth-first processing
            task_spec = self.spatial_specs[task.level]
            # Collect and emit a fraction of these, according to the desired probability
            p = prob_per_level[task.level]
            emitted_subset = []
            for i in range(len(task.annotations) - 1, -1, -1):
                if random() > p:
                    continue
                emitted_subset.append(task.annotations[i])
                del task.annotations[i]
            file_path = path_join(dir_path, task_spec.key, task.file_name())
            if task_spec.sharding is None:
                self.write_annotations(file_path, emitted_subset, True)
            else:
                chunk_data = self.compile_multi_annotation_buffer(emitted_subset, True)
                chunk_id = compressed_morton_code(task.cell_index, task_spec.grid_shape)
                chunks_per_level[task.level].append(Chunk(chunk_id, chunk_data))
            emitted_count = len(emitted_subset)

            self.spatial_specs[task.level].limit = max(
                self.spatial_specs[task.level].limit, emitted_count
            )

            # Now, subdivide remaining annotations into sub-tasks.
            if task.level < max_level:
                # Get the next level's spatial spec
                next_level = task.level + 1
                next_spec = self.spatial_specs[next_level]

                # Get the range of child cells for this parent cell
                child_ranges = get_child_cell_ranges(
                    self.spatial_specs, task.level, task.cell_index
                )

                # Iterate over child grid cells within the calculated ranges
                for x in range(child_ranges[0][0], child_ranges[0][1]):
                    for y in range(child_ranges[1][0], child_ranges[1][1]):
                        for z in range(child_ranges[2][0], child_ranges[2][1]):
                            child_cell_index = (x, y, z)

                            # Calculate the bounding box for this child cell
                            start = [
                                self.lower_bound[0] + x * next_spec.chunk_size[0],
                                self.lower_bound[1] + y * next_spec.chunk_size[1],
                                self.lower_bound[2] + z * next_spec.chunk_size[2],
                            ]
                            end = [
                                start[0] + next_spec.chunk_size[0],
                                start[1] + next_spec.chunk_size[1],
                                start[2] + next_spec.chunk_size[2],
                            ]

                            bbox = BBox3D.from_coords(start, end)

                            # Find annotations that intersect this child cell
                            child_annotations = [
                                annotation
                                for annotation in task.annotations
                                if annotation.in_bounds(bbox)
                            ]

                            # Only create a task if there are annotations in this cell
                            if not child_annotations:
                                continue

                            child_task = SubdivideTask(
                                level=next_level,
                                cell_index=child_cell_index,
                                annotations=child_annotations,
                            )
                            taskQ.append(child_task)

        # Now, if we have collected any chunks for sharding, write those out now!
        for chunks, spec in zip(chunks_per_level, self.spatial_specs):
            if not chunks:
                continue
            shard_dir = path_join(dir_path, spec.key)
            write_shard_files(shard_dir, spec.sharding, chunks)
            print(f"Wrote {len(chunks)} chunks to shards in {shard_dir}")


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
