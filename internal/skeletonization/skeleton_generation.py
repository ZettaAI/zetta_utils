# pylint: disable = c-extension-no-member
from __future__ import annotations

from typing import Sequence

from cloudvolume import CloudVolume

from zetta_utils import builder, log, mazepa
from zetta_utils.geometry.bbox import BBox3D
from zetta_utils.layer.db_layer import DBLayer
from zetta_utils.layer.volumetric.layer import VolumetricLayer
from zetta_utils.mazepa_layer_processing.common.subchunkable_apply_flow import (
    build_subchunkable_apply_flow,
)

from .fragments import MakeSkeletonFragsOperation
from .shards import MakeSkeletonShardsFlowSchema

logger = log.get_logger("zetta_utils")


@builder.register("build_generate_skeletons_flow")
def build_generate_skeletons_flow(  # pylint: disable=too-many-locals, dangerous-default-value # teasar-params
    # Common arguments
    segmentation: VolumetricLayer,
    seg_db: DBLayer,
    frag_db: DBLayer,
    skeleton_dir: str,
    # Fragment arguments, except for frag_path
    seg_resolution: Sequence[float],
    frag_chunk_size: Sequence[int],
    frag_path: str | None = None,
    expand_bbox_resolution: bool = False,
    expand_bbox_processing: bool = True,
    bbox: BBox3D | None = None,
    start_coord: Sequence[int] | None = None,
    end_coord: Sequence[int] | None = None,
    coord_resolution: Sequence | None = None,
    auto_bbox: bool = False,
    frag_teasar_params: dict[str, int] = {"scale": 10, "const": 10},
    frag_high_padding: int = 1,
    frag_low_padding: int = 0,
    frag_object_ids: list | None = None,
    frag_mask_ids: list | None = None,
    frag_cross_sectional_area: bool = False,
    frag_cross_sectional_area_smoothing_window: int = 1,
    frag_cross_sectional_area_shape_delta: int = 150,
    frag_strip_integer_attributes: bool = True,
    frag_fix_branching: bool = True,
    frag_fix_borders: bool = True,
    frag_fix_avocados: bool = False,
    frag_fill_holes: bool = False,
    frag_dust_threshold: int = 1000,
    # Sharding arguments
    shard_index_bytes: int = 2 ** 13,
    minishard_index_bytes: int = 2 ** 15,
    min_shards: int = 1,
    num_shard_no_tasks: int = 1,
    minishard_index_encoding: str = "gzip",
    max_seg_ids_per_shard: int | None = None,
    shard_data_encoding: str = "gzip",
    shard_max_cable_length: float | None = None,
    shard_dust_threshold: int = 4000,
    shard_tick_threshold: int = 6000,
) -> mazepa.Flow:

    """

    Generates a sharded skeletonisation from a segmentation. The
    generation happens in two parts:

    1) The bounding box is turned chunkwise into skeleton fragments that
    contain the part of each segment ID's skeleton within the chunk. The
    record of these mesh fragments is written to two different databases.
    2) For each shard in the sharded mesh, the databases are used to figure out
    what segment IDs should be in the shard, as well as which fragment the segment ID
    belongs to. Each shard is then generated.


    to be densely skeletonized. The default shape (512,512,512)
    was designed to work within 6 GB of RAM on average at parallel=1
    but can exceed this amount for certain objects such as glia.
    4 GB is usually OK.

    :param segmentation: The VolumetricLayer containing the segmentation to use.
    :param seg_db: The DBLayer that contains each segment id - only used to keep
        track of the number of segments (necessary for sharding parameter computation).
    :param frag_db: The DBLayer that contains each segment id and fragment name, for
        all segids in each fragment. This database is used to look up fragments when
        sharding a given segid during sharding.
    :param skeleton_dir: The mesh directory for the run; the fragments will be in
        `{frag_path}/{skeleton_dir}`, and the shards will be in `{segmentation_path}/{mesh_dir}`.
    :param seg_resolution: Resolution of the segmentation to use for generating fragments.
    :param frag_chunk_size: The size of each fragment chunk. Must evenly divide the ``bbox``
        given (whether given as a ``bbox`` or ``start_coord``, ``end_coord``, and
        ``coord_resolution``) if ``expand_bbox_processing`` is `False`.
    :param frag_path: Where to store the fragments (with the `mesh_dir` prefix on
        the filenames). If not provided, defaults to the segmentation folder.
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
    :param frag_mask_ids: If provided, ignore these ids.
    :param frag_teasar_params:
        NOTE: see github.com/seung-lab/kimimaro for an updated list
            see https://github.com/seung-lab/kimimaro/wiki/\
                Intuition-for-Setting-Parameters-const-and-scale
            for help with setting these parameters.
        NOTE: DBF = Distance from Boundary Field (i.e. euclidean distance transform)

        scale: float, multiply invalidation radius by distance from boundary
        const: float, add this physical distance to the invalidation radius
        soma_detection_threshold: if object has a DBF value larger than this,
            root will be placed at largest DBF value and special one time invalidation
            will be run over that root location (see soma_invalidation scale)
            expressed in chosen physical units (i.e. nm)
        pdrf_scale: scale factor in front of dbf, used to weight DBF over euclidean distance
            (higher to pay more attention to dbf)
        pdrf_exponent: exponent in dbf formula on distance from edge, faster if factor of 2
            (default 16)
        soma_invalidation_scale: the 'scale' factor used in the one time soma root invalidation
            (default .5)
        soma_invalidation_const: the 'const' factor used in the one time soma root invalidation
            (default 0)
    :param frag_fix_branching: Trades speed for quality of branching at forks. You'll
        almost always want this set to True.
    :param frag_fix_borders: Allows trivial merging of single overlap tasks. You'll only
        want to set this to false if you're working on single or non-overlapping
        volumes.
    :param frag_fix_avocados: Fixes artifacts from oversegmented cell somata.
    :param frag_fix_holes: Removes input labels that are deemed to be holes.
    :param frag_dust_threshold: Don't skeletonize labels smaller than this number of voxels
        as seen by a single task.
    :param frag_cross_sectional_area: At each vertex, compute the area covered by a
    section plane whose direction is defined by the normal vector pointing
    to the next vertex in the sequence. (n.b. this will add significant time
    to the total computation.)
    :param frag_cross_sectional_area_smoothing_window: Perform a rolling average of the
        normal vectors across these many vectors.
    :param frag_cross_sectional_area_shape_delta: See kimimaro documentation.
    :param shard_index_bytes: Sharding parameter; see `_compute_shard_params_for_hashed`.
    :param minishard_index_bytes: Sharding parameter; see `_compute_shard_params_for_hashed`.
    :param min_shards: Sharding parameter; see `_compute_shard_params_for_hashed`.
    :param num_shard_no_tasks: Number of tasks to generate for assigning shard numbers
        to mesh fragments in the `frag_db`.
    :param num_lods: Number of LODs (mesh equivalent of MIPs) to generate.
    :param shard_draco_compression_level: Draco compression level.
    :param minishard_index_encoding: Minishard index encoding; see
        "https://github.com/seung-lab/igneous/blob/
        7258212e57c5cfc1e5f0de8162f830c7d49e1be9/igneous/task_creation/image.py#L343"
        for details.
    :param max_seg_ids_per_shard: Max number of segment ids that can be assigned
        to a single shard.
    :param data_encoding: Encoding for shard files.
    :param shard_max_cable_length: Sharding parameter, see kimimaro documentation.
    :param shard_dust_threshold: Skip segments smaller than this number in voxels.
    :param shard_tick_threshold: Keep ticks larger than this number in nanometres.
    """

    segmentation_path = segmentation.name
    cv = CloudVolume(segmentation.name, mip=list(seg_resolution))

    skeleton_dir_to_use = skeleton_dir or cv.info.get("skeletons", None)
    if skeleton_dir_to_use is None:
        raise ValueError(
            f"`skeleton_dir` has not been specified, but the infofile at {segmentation_path} "
            "does not specify the skeleton directory."
        )

    if not "skeletons" in cv.info:
        cv.info["skeletons"] = skeleton_dir_to_use
        cv.commit_info()
    else:
        if not cv.info["skeletons"] == skeleton_dir:
            raise ValueError(
                f"`skeleton_dir` has been specified as {skeleton_dir}, but the infofile at "
                f"{segmentation_path} specifies a different directory, {cv.info['skeletons']}"
            )

    cv.skeleton.meta.info["mip"] = int(cv.meta.to_mip(seg_resolution))
    cv.skeleton.meta.info["vertex_attributes"] = [
        attr
        for attr in cv.skeleton.meta.info["vertex_attributes"]
        if attr["data_type"] == "float32"
    ]

    if frag_cross_sectional_area:
        has_cross_sectional_area_attr = any(
            attr["id"] == "cross_sectional_area"
            for attr in cv.skeleton.meta.info["vertex_attributes"]
        )
        if not has_cross_sectional_area_attr:
            cv.skeleton.meta.info["vertex_attributes"].append(
                {
                    "id": "cross_sectional_area",
                    "data_type": "float32",
                    "num_components": 1,
                }
            )
    cv.skeleton.meta.commit_info()

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
        allow_cache_up_to_level=None,
        print_summary=True,
        generate_ng_link=False,
        op=MakeSkeletonFragsOperation((0, 0, 0)),
        op_args=(),
        op_kwargs={
            "segmentation": segmentation,
            "seg_db": seg_db,
            "frag_db": frag_db,
            "skeleton_dir": skeleton_dir,
            "frag_path": frag_path,
            "high_padding": frag_high_padding,
            "low_padding": frag_low_padding,
            "object_ids": frag_object_ids,
            "mask_ids": frag_mask_ids,
            "teasar_params": frag_teasar_params,
            "cross_sectional_area": frag_cross_sectional_area,
            "cross_sectional_area_smoothing_window": frag_cross_sectional_area_smoothing_window,
            "cross_sectional_area_shape_delta": frag_cross_sectional_area_shape_delta,
            "strip_integer_attributes": frag_strip_integer_attributes,
            "fix_branching": frag_fix_branching,
            "fix_borders": frag_fix_borders,
            "fix_avocados": frag_fix_avocados,
            "fill_holes": frag_fill_holes,
            "dust_threshold": frag_dust_threshold,
        },
        auto_bbox=auto_bbox,
        bbox=bbox,
        start_coord=start_coord,
        end_coord=end_coord,
        coord_resolution=coord_resolution,
    )
    sharding_flow = MakeSkeletonShardsFlowSchema()(
        segmentation_path=segmentation_path,
        seg_db=seg_db,
        frag_db=frag_db,
        shard_index_bytes=shard_index_bytes,
        minishard_index_bytes=minishard_index_bytes,
        min_shards=min_shards,
        num_shard_no_tasks=num_shard_no_tasks,
        minishard_index_encoding=minishard_index_encoding,
        skeleton_dir=skeleton_dir,
        frag_path=frag_path,
        max_seg_ids_per_shard=max_seg_ids_per_shard,
        data_encoding=shard_data_encoding,
        max_cable_length=shard_max_cable_length,
        dust_threshold=shard_dust_threshold,
        tick_threshold=shard_tick_threshold,
    )
    return mazepa.sequential_flow([fragments_flow, sharding_flow])
