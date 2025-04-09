# pylint: disable = c-extension-no-member
from __future__ import annotations

import os
from typing import Sequence, Tuple

from cloudfiles import CloudFiles

from zetta_utils import builder, log, mazepa
from zetta_utils.geometry.bbox import BBox3D
from zetta_utils.layer.db_layer.firestore import temp_firestore_layer_ctx
from zetta_utils.layer.volumetric.layer import VolumetricLayer
from zetta_utils.mazepa.flows import FlowFnReturnType, flow_schema_cls
from zetta_utils.mazepa_layer_processing.common.subchunkable_apply_flow import (
    build_subchunkable_apply_flow,
)

from .fragments import MakeMeshFragsOperation
from .shards import MakeMeshShardsFlowSchema

logger = log.get_logger("zetta_utils")


@flow_schema_cls
class GenerateMeshesFlowSchema:
    def flow(  # pylint: disable=no-self-use, too-many-locals, line-too-long
        self,
        segmentation: VolumetricLayer,
        mesh_dir: str,
        seg_resolution: Sequence[float],
        frag_chunk_size: Sequence[int],
        frag_path: str | None,
        overwrite_meshes: bool,
        expand_bbox_resolution: bool,
        expand_bbox_processing: bool,
        bbox: BBox3D | None,
        start_coord: Sequence[int] | None,
        end_coord: Sequence[int] | None,
        coord_resolution: Sequence | None,
        auto_bbox: bool,
        frag_high_padding: int,
        frag_low_padding: int,
        frag_simplification_factor: int,
        frag_max_simplification_error: int,
        frag_draco_compression_level: int,
        frag_draco_create_metadata: bool,
        frag_object_ids: list | None,
        frag_closed_dataset_edges: bool,
        frag_num_splits: Sequence[int],
        shard_index_bytes: int,
        minishard_index_bytes: int,
        min_shards: int,
        num_frags_per_shard_no_task: int,
        num_lods: int,
        shard_draco_compression_level: int,
        vertex_quantization_bits: int,
        minishard_index_encoding: str,
        min_chunk_size: Tuple[int, int, int],
        max_seg_ids_per_shard: int | None,
        worker_type: str | None,
    ) -> FlowFnReturnType:
        """
        Generates a Draco compressed sharded mesh from a segmentation. The generation
        happens in two parts:
        1) The bounding box is turned chunkwise into Draco compressed mesh fragments that
        contain the part of each segment ID's mesh within the chunk, with optional
        subchunking. The record of these mesh fragments is written to two different databases.
        2) For each shard in the sharded mesh, the databases are used to figure out
        what segment IDs should be in the shard, as well as which fragment the segment ID
        belongs to. Each shard is then generated.

        :param mesh_dir: The mesh directory for the run; the fragments will be in
            `{frag_path}/{mesh_dir}`, and the shards will be in `{segmentation_path}/{mesh_dir}`.
            Note that this operation does NOT modify the infofile of the segmentation with the
            `mesh_dir` information - this is handled during sharding.
            MUST START WITH `mesh_mip_[0-9]+` FOR SHARDING TO DETECT MIP, WHICH IS RELATIVE TO
            THE SEGMENTATION INFOFILE.
        :param seg_resolution: Resolution of the segmentation to use for generating fragments.
        :param frag_chunk_size: The size of each fragment chunk. Must evenly divide the ``bbox``
            given (whether given as a ``bbox`` or ``start_coord``, ``end_coord``, and
            ``coord_resolution``) if ``expand_bbox_processing`` is `False`.
        :param frag_path: Where to store the fragments (with the `mesh_dir` prefix on
            the filenames). If not provided, defaults to the segmentation folder.
        :param overwrite_meshes: Whether to overwrite existing meshes. If `True`, the operation
            will delete any existing meshes and fragments in the `mesh_dir` directory. If `False`,
            the operation will check if the meshes and fragments already exist and raise an Error
            if so.
        :param expand_bbox_resolution: Expands ``bbox`` (whether given as a ``bbox`` or
            ``start_coord``, ``end_coord``, and ``coord_resolution``) to be integral in the
            ``seg_resolution``.
        :param expand_bbox_processing: Expands ``bbox`` (whether given as a ``bbox`` or
            ``start_coord``, ``end_coord``, and ``coord_resolution``) to be an integral
            multiple of ``frag_chunk_size``.
        :param bbox: The bounding box for the fragmentation. Cannot be used with ``start_coord``,
            ``end_coord``, and ``coord_resolution``, or ``auto_bbox``.
        :param start_coord: The start coordinate for the bounding box. Must be used with
            ``end_coord`` and ``coord_resolution``; cannot be used with ``bbox`` or ``auto_bbox.
        :param end_coord: The end coordinate for the bounding box. Must be used with ``start_coord``
            and ``coord_resolution``; cannot be used with ``bbox`` or ``auto_bbox.
        :param coord_resolution: The resolution in which the coordinates are given for the bounding
            box. Must be used with ``start_coord`` and ``end_coord``; cannot be used with ``bbox``.
        :param auto_bbox: Sets the ``bbox`` to the bounds of ``segmentation`` at ``seg_resolution``.
            Note that this may result in a significant overhead if the ``segmentation`` bounds have
            not been tightened. Cannot be used with ``bbox`` or ``start_coord``, ``end_coord``, and
            ``coord_resolution``.
        :param frag_high_padding: Padding on the high side; recommended to keep default.
        :param frag_low_padding: Padding on the low side; recommended to keep default.
        :param frag_simplification factor: What factor to try to reduce the number of triangles
            in the mesh, constrained by `max_simplification_error`.
        :param frag_max_simplification_error: The maximum physical distance that
            simplification is allowed to move a triangle vertex by.
        :param frag_draco_compression_level: Draco compression level.
        :param frag_draco_create_metadata: Whether to create Draco metadata.
        :param frag_object_ids: If provided, only generate fragments for these ids.
        :param frag_closed_dataset_edges: Close the mesh faces at the edge of the dataset.
        :param frag_num_splits: Split the `idx` into `(NUM_X, NUM_Y, NUM_Z)` chunks when
        meshing. zmesh uses double the memory if the size of an object exceeds
        (1023, 1023, 511) in any axis due to the 32-bit limitation, so dividing
        the idx and processing in chunks keeps the size of the objects below
        this limit and reduces the number of fragments.
        :param shard_index_bytes: Sharding parameter; see `_compute_shard_params_for_hashed`.
        :param minishard_index_bytes: Sharding parameter; see `_compute_shard_params_for_hashed`.
        :param min_shards: Sharding parameter; see `_compute_shard_params_for_hashed`.
        :param num_frags_per_shard_no_task: Number of tasks to generate for assigning shard numbers
            to mesh fragments in the `frag_db`.
        :param num_lods: Number of LODs (mesh equivalent of MIPs) to generate.
        :param shard_draco_compression_level: Draco compression level.
        :param minishard_index_encoding: Minishard index encoding; see
            "https://github.com/seung-lab/igneous/blob/
            7258212e57c5cfc1e5f0de8162f830c7d49e1be9/igneous/task_creation/image.py#L343"
            for details.
        :param max_seg_ids_per_shard: Max number of segment ids that can be assigned
            to a single shard.
        :param min_chunk_size: Minimum chunk size for a mesh to be split during
            LOD generation.
        """
        cf = CloudFiles(os.path.join(segmentation.name, mesh_dir))
        mesh_dir_contents = list(cf.list())
        if overwrite_meshes:
            cf.delete(mesh_dir_contents)
        else:
            if any(fn.endswith(".frags") or fn.endswith(".shard") for fn in mesh_dir_contents):
                raise FileExistsError(
                    f"Mesh directory `{mesh_dir}` in `{segmentation.name}` has existing "
                    "fragments and/or shards but `overwrite_meshes` is False."
                )
        with temp_firestore_layer_ctx("seg_db") as seg_db, temp_firestore_layer_ctx(
            "frag_db"
        ) as frag_db:
            fragments_flow = build_subchunkable_apply_flow(
                dst=segmentation,
                dst_resolution=seg_resolution,
                processing_chunk_sizes=[frag_chunk_size],
                processing_gap=(0, 0, 0),
                processing_crop_pads=(0, 0, 0),
                processing_blend_pads=(0, 0, 0),
                processing_blend_modes="quadratic",
                level_intermediaries_dirs=None,
                skip_intermediaries=True,
                max_reduction_chunk_size=None,
                expand_bbox_resolution=expand_bbox_resolution,
                expand_bbox_processing=expand_bbox_processing,
                shrink_processing_chunk=False,
                auto_divisibility=False,
                op_worker_type=worker_type,
                allow_cache_up_to_level=None,
                print_summary=True,
                generate_ng_link=False,
                op=MakeMeshFragsOperation((0, 0, 0)),
                op_args=(),
                op_kwargs={
                    "segmentation": segmentation,
                    "seg_db": seg_db,
                    "frag_db": frag_db,
                    "mesh_dir": mesh_dir,
                    "frag_path": frag_path,
                    "high_padding": frag_high_padding,
                    "low_padding": frag_low_padding,
                    "simplification_factor": frag_simplification_factor,
                    "max_simplification_error": frag_max_simplification_error,
                    "draco_compression_level": frag_draco_compression_level,
                    "draco_create_metadata": frag_draco_create_metadata,
                    "object_ids": frag_object_ids,
                    "closed_dataset_edges": frag_closed_dataset_edges,
                    "num_splits": frag_num_splits,
                },
                auto_bbox=auto_bbox,
                bbox=bbox,
                start_coord=start_coord,
                end_coord=end_coord,
                coord_resolution=coord_resolution,
            )
            segmentation_path = segmentation.name
            sharding_flow = MakeMeshShardsFlowSchema()(
                segmentation_path=segmentation_path,
                seg_db=seg_db,
                frag_db=frag_db,
                shard_index_bytes=shard_index_bytes,
                minishard_index_bytes=minishard_index_bytes,
                min_shards=min_shards,
                num_frags_per_shard_no_task=num_frags_per_shard_no_task,
                num_lods=num_lods,
                draco_compression_level=shard_draco_compression_level,
                vertex_quantization_bits=vertex_quantization_bits,
                minishard_index_encoding=minishard_index_encoding,
                mesh_dir=mesh_dir,
                frag_path=frag_path,
                min_chunk_size=min_chunk_size,
                max_seg_ids_per_shard=max_seg_ids_per_shard,
                worker_type=worker_type,
            )
            yield fragments_flow
            yield mazepa.Dependency()
            yield sharding_flow
            yield mazepa.Dependency()


@builder.register("build_generate_meshes_flow")
def build_generate_meshes_flow(  # pylint: disable=too-many-locals
    segmentation: VolumetricLayer,
    mesh_dir: str,
    seg_resolution: Sequence[float],
    frag_chunk_size: Sequence[int],
    frag_path: str | None = None,
    overwrite_meshes: bool = False,
    expand_bbox_resolution: bool = False,
    expand_bbox_processing: bool = True,
    bbox: BBox3D | None = None,
    start_coord: Sequence[int] | None = None,
    end_coord: Sequence[int] | None = None,
    coord_resolution: Sequence | None = None,
    auto_bbox: bool = False,
    frag_high_padding: int = 1,
    frag_low_padding: int = 0,
    frag_simplification_factor: int = 100,
    frag_max_simplification_error: int = 40,
    frag_draco_compression_level: int = 1,
    frag_draco_create_metadata: bool = False,
    frag_object_ids: list | None = None,
    frag_closed_dataset_edges: bool = True,
    frag_num_splits: Sequence[int] = (1, 1, 1),
    shard_index_bytes: int = 2 ** 13,
    minishard_index_bytes: int = 2 ** 15,
    min_shards: int = 1,
    num_frags_per_shard_no_task: int = 1024,
    num_lods: int = 0,
    shard_draco_compression_level: int = 7,
    vertex_quantization_bits: int = 16,
    minishard_index_encoding: str = "gzip",
    min_chunk_size: Tuple[int, int, int] = (256, 256, 256),
    max_seg_ids_per_shard: int | None = None,
    worker_type: str | None = None,
) -> mazepa.Flow:
    return GenerateMeshesFlowSchema()(
        segmentation=segmentation,
        mesh_dir=mesh_dir,
        seg_resolution=seg_resolution,
        frag_chunk_size=frag_chunk_size,
        frag_path=frag_path,
        overwrite_meshes=overwrite_meshes,
        expand_bbox_resolution=expand_bbox_resolution,
        expand_bbox_processing=expand_bbox_processing,
        bbox=bbox,
        start_coord=start_coord,
        end_coord=end_coord,
        coord_resolution=coord_resolution,
        auto_bbox=auto_bbox,
        frag_high_padding=frag_high_padding,
        frag_low_padding=frag_low_padding,
        frag_simplification_factor=frag_simplification_factor,
        frag_max_simplification_error=frag_max_simplification_error,
        frag_draco_compression_level=frag_draco_compression_level,
        frag_draco_create_metadata=frag_draco_create_metadata,
        frag_object_ids=frag_object_ids,
        frag_closed_dataset_edges=frag_closed_dataset_edges,
        frag_num_splits=frag_num_splits,
        shard_index_bytes=shard_index_bytes,
        minishard_index_bytes=minishard_index_bytes,
        min_shards=min_shards,
        num_frags_per_shard_no_task=num_frags_per_shard_no_task,
        num_lods=num_lods,
        shard_draco_compression_level=shard_draco_compression_level,
        vertex_quantization_bits=vertex_quantization_bits,
        minishard_index_encoding=minishard_index_encoding,
        min_chunk_size=min_chunk_size,
        max_seg_ids_per_shard=max_seg_ids_per_shard,
        worker_type=worker_type,
    )
