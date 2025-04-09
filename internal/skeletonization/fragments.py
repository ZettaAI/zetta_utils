from __future__ import annotations

import uuid
from typing import Sequence

import attrs
import fastremap
import kimimaro
import numpy as np
from cloudfiles import CloudFiles
from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox
from dbscan import DBSCAN  # pylint: disable=no-name-in-module
from mapbuffer import MapBuffer

from zetta_utils import builder, log, mazepa
from zetta_utils.geometry import Vec3D
from zetta_utils.layer.db_layer import DBLayer
from zetta_utils.layer.volumetric import VolumetricIndex, VolumetricLayer

# from zetta_utils.mazepa import semaphore

logger = log.get_logger("zetta_utils")

## Most of this file is a direct port of the
## sharded skeleton generation in igneous to zetta_utils:
## https://github.com/seung-lab/igneous


@builder.register("MakeSkeletonFragsOperation")
@mazepa.taskable_operation_cls
@attrs.frozen()
class MakeSkeletonFragsOperation:  # pylint: disable=no-self-use
    """
    Stage 1 of skeletonization.

    Convert chunks of segmentation into chunked skeletons and point clouds.
    Must be followed up by sharding.

    Assign tasks with one voxel overlap in a regular grid
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
    :param frag_path: Where to store the fragments (with the `mesh_dir` prefix on
        the filenames). If not provided, defaults to the segmentation folder.
    :param expand_bbox_resolution: Expands ``bbox`` (whether given as a ``bbox`` or
        ``start_coord``, ``end_coord``, and ``coord_resolution``) to be integral in the
        ``seg_resolution``.
    :param expand_bbox_processing: Expands ``bbox`` (whether given as a ``bbox`` or
        ``start_coord``, ``end_coord``, and ``coord_resolution``) to be an integral
        multiple of ``frag_chunk_size``.
    :param high_padding: Padding on the high side; recommended to keep default.
    :param low_padding: Padding on the low side; recommended to keep default.
    :param simplification factor: What factor to try to reduce the number of triangles
        in the mesh, constrained by `max_simplification_error`.
    :param max_simplification_error: The maximum physical distance that
        simplification is allowed to move a triangle vertex by.
    :param draco_compression_level: Draco compression level.
    :param draco_create_metadata: Whether to create Draco metadata.
    :param object_ids: If provided, only generate fragments for these ids.
    :param mask_ids: If provided, ignore these ids.
    :param teasar_params:
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
    :param fix_branching: Trades speed for quality of branching at forks. You'll
        almost always want this set to True.
    :param fix_borders: Allows trivial merging of single overlap tasks. You'll only
        want to set this to false if you're working on single or non-overlapping
        volumes.
    :param fix_avocados: Fixes artifacts from oversegmented cell somata.
    :param fix_holes: Removes input labels that are deemed to be holes.
    :param dust_threshold: Don't skeletonize labels smaller than this number of voxels
        as seen by a single task.
    :param cross_sectional_area: At each vertex, compute the area covered by a
    section plane whose direction is defined by the normal vector pointing
    to the next vertex in the sequence. (n.b. this will add significant time
    to the total computation.)
    :param cross_sectional_area_smoothing_window: Perform a rolling average of the
        normal vectors across these many vectors.
    :param cross_sectional_area_shape_delta: See kimimaro documentation.
    :param parallel: number of processes to deploy against a single task. parallelizes
        over labels, it won't speed up a single complex label. You can be slightly
        more memory efficient using a single big task with parallel than with seperate
        tasks that add up to the same volume. Unless you know what you're doing, stick
        with parallel=1 for cloud deployments.
    """

    crop_pad: Sequence[int] = (0, 0, 0)

    def get_operation_name(self):
        return "MakeSkeletonFragsOperation"

    def get_input_resolution(self, dst_resolution: Vec3D[float]) -> Vec3D[float]:
        return dst_resolution  # TODO add support for data res

    def with_added_crop_pad(self, crop_pad: Vec3D[int]) -> MakeSkeletonFragsOperation:
        return attrs.evolve(self, crop_pad=Vec3D(*self.crop_pad) + crop_pad)

    def __call__(  # pylint: disable=too-many-locals
        self,
        idx: VolumetricIndex,
        dst: VolumetricLayer,
        segmentation: VolumetricLayer,
        seg_db: DBLayer,
        frag_db: DBLayer,
        teasar_params: dict,
        skeleton_dir: str,
        object_ids: Sequence[int] | None = None,
        mask_ids: Sequence[int] | None = None,
        frag_path: str | None = None,
        cross_sectional_area: bool = False,
        cross_sectional_area_smoothing_window: int = 1,
        cross_sectional_area_shape_delta: int = 150,
        high_padding: int = 1,
        low_padding: int = 0,
        strip_integer_attributes: bool = True,
        fix_branching: bool = True,
        fix_borders: bool = True,
        fix_avocados: bool = False,
        fill_holes: bool = False,
        dust_threshold: int = 1000,
        parallel: int = 1,
    ):

        # Marching cube needs 1vx overlaps to not have lines appear
        # between adjacent chunks.
        idx_padded = idx.translated_start(
            (-low_padding, -low_padding, -low_padding)
        ).translated_end((high_padding, high_padding, high_padding))

        idx_padded_full = idx_padded.intersection(segmentation.backend.get_bounds(idx.resolution))

        all_labels = segmentation[idx_padded_full][0]

        if mask_ids:
            all_labels = fastremap.mask(all_labels, mask_ids)

        # TODO implement synapses as DBLayer
        """
        extra_targets_after = {}

        if synapses:
            centroids, kdtree, labelsmap = _synapses_in_space(synapses, N=len(num_synapses))
            extra_targets_after = kimimaro.synapses_to_targets(all_labels, synapses_in_spase)
        """
        skeletons = kimimaro.skeletonize(
            all_labels,
            teasar_params,
            object_ids=object_ids,
            anisotropy=idx_padded_full.resolution,
            dust_threshold=dust_threshold,
            progress=False,
            fix_branching=fix_branching,
            fix_borders=fix_borders,
            fix_avocados=fix_avocados,
            fill_holes=fill_holes,
            parallel=parallel,
            # extra_targets_after=extra_targets_after.keys(), # TODO implement synapses as DBLayer
        )
        del all_labels

        if len(skeletons) == 0:
            return
        if cross_sectional_area:  # This is expensive!
            skeletons = _compute_cross_sectional_area(
                segmentation,
                idx_padded_full,
                skeletons,
                cross_sectional_area_shape_delta,
                cross_sectional_area_smoothing_window,
                fill_holes,
                mask_ids,
            )

        # TODO: Reimplement half pixel correction - our bbox updates are not guaranteed
        # have correct offsets for other mips
        # voxel centered (+0.5) and uses more accurate bounding box from mip 0
        # corrected_offset = (bbox.minpt.astype(np.float32) - vol.meta.voxel_offset(self.mip) + 0.5) * vol.meta.resolution(self.mip) # pylint: disable=line-too-long
        # corrected_offset += vol.meta.voxel_offset(0) * vol.meta.resolution(0)
        corrected_offset = idx_padded_full.resolution * idx_padded_full.start
        for _, skel in skeletons.items():
            skel.vertices[:] += corrected_offset

        # TODO implement synapses as DBLayer
        """
        if synapses:
            for skel in skeletons.values():
                terminal_nodes = skel.vertices[skel.terminals()]

                for i, vert in enumerate(terminal_nodes):
                    vert_moved = Vec3D(*vert) / idx.resolution - idx.start
                    vert_int = tuple(round(vert_moved).int())
                    if vert_int in extra_targets_after:
                        skel.vertex_types[i] = extra_targets_after[vert]
        """

        # old versions of neuroglancer don't
        # support int attributes
        if strip_integer_attributes:
            _strip_integer_attributes(skeletons.values())

        _upload_batch(
            segmentation, skeleton_dir, frag_path, idx_padded_full, frag_db, seg_db, skeletons
        )


def _strip_integer_attributes(skeletons):
    for skel in skeletons:
        skel.extra_attributes = [
            attr for attr in skel.extra_attributes if attr["data_type"] in ("float32", "float64")
        ]
    return skeletons


def _compute_cross_sectional_area(
    segmentation: VolumetricLayer,
    idx_padded_full: VolumetricIndex,
    skeletons,
    cross_sectional_area_shape_delta: int,
    cross_sectional_area_smoothing_window: int,
    fill_holes: bool,
    mask_ids: Sequence[int] | None = None,
):
    # Why redownload a bigger image? In order to avoid clipping the
    # cross sectional areas on the edges.

    vol = CloudVolume(segmentation.name, mip=list(idx_padded_full.resolution))
    bbox = Bbox(idx_padded_full.start, idx_padded_full.stop)

    delta = cross_sectional_area_shape_delta

    big_bbox = bbox.clone()
    big_bbox.grow(delta)
    big_bbox = Bbox.clamp(big_bbox, vol.bounds)

    huge_bbox = big_bbox.clone()
    huge_bbox.grow(int(np.max(bbox.size()) / 2) + 1)
    huge_bbox = Bbox.clamp(huge_bbox, vol.bounds)

    mem_vol = vol.image.memory_cutout(huge_bbox, mip=vol.mip, encoding="crackle", compress=False)

    all_labels = mem_vol[big_bbox][..., 0]

    delta = bbox.minpt - big_bbox.minpt

    # place the skeletons in exactly the same position
    # in the enlarged image
    for skel in skeletons.values():
        skel.vertices += delta * vol.resolution

    if mask_ids:
        all_labels = fastremap.mask(all_labels, mask_ids)

    skeletons = kimimaro.cross_sectional_area(
        all_labels,
        skeletons,
        anisotropy=vol.resolution,
        smoothing_window=cross_sectional_area_smoothing_window,
        progress=False,
        in_place=True,
        fill_holes=fill_holes,
    )

    del all_labels

    # move the vertices back to their old smaller image location
    for skel in skeletons.values():
        skel.vertices -= delta * vol.resolution

    return _repair_cross_sectional_area_contacts(
        mem_vol,
        bbox,
        skeletons,
        cross_sectional_area_shape_delta,
        cross_sectional_area_smoothing_window,
        fill_holes,
    )


def _repair_cross_sectional_area_contacts(
    vol: CloudVolume,
    bbox: Bbox,
    skeletons,
    cross_sectional_area_shape_delta: int,
    cross_sectional_area_smoothing_window: int,
    fill_holes: bool,
):

    repair_skels = [
        skel for skel in skeletons.values() if np.any(skel.cross_sectional_area_contacts > 0)
    ]

    delta = int(cross_sectional_area_shape_delta)

    shape = bbox.size3()

    def reprocess_skel(pts, skel):
        pts_bbx = Bbox.from_points(pts)

        pts_bbx_vol = pts_bbx + bbox.minpt
        center = pts_bbx_vol.center().astype(int)
        skel_bbx = Bbox(center, center + 1)
        skel_bbx.grow(delta + shape // 2)

        skel_bbx = Bbox.clamp(skel_bbx, vol.bounds)

        binary_image = vol.download(skel_bbx, mip=vol.mip, label=skel.id)[..., 0]

        diff = bbox.minpt - skel_bbx.minpt
        skel.vertices += diff * vol.resolution

        kimimaro.cross_sectional_area(
            binary_image,
            skel,
            anisotropy=vol.resolution,
            smoothing_window=cross_sectional_area_smoothing_window,
            progress=False,
            in_place=True,
            fill_holes=fill_holes,
            repair_contacts=True,
        )

        skel.vertices -= diff * vol.resolution

    for skel in repair_skels:
        verts = (skel.vertices // vol.resolution).astype(int)
        reprocess_skel(verts, skel)

        pts = verts[skel.cross_sectional_area_contacts > 0]
        if len(pts) == 0:
            continue

        labels, core_samples_mask = DBSCAN(  # pylint: disable=unused-variable
            pts, eps=5, min_samples=2
        )
        uniq = fastremap.unique(labels)
        for lbl in uniq:
            reprocess_skel(pts[labels == lbl], skel)

    return skeletons


def _upload_batch(
    segmentation: VolumetricLayer,
    skeleton_dir: str,
    frag_path: str | None,
    idx: VolumetricIndex,
    frag_db: DBLayer,
    seg_db: DBLayer,
    skeletons,
):
    mbuf = MapBuffer(skeletons, compress="br", tobytesfn=lambda skel: skel.to_precomputed())

    frag_path_to_use = frag_path or segmentation.name
    cf = CloudFiles(frag_path_to_use)
    cf.put(
        path=f"{skeleton_dir}/{Bbox(idx.start, idx.stop).to_filename()}.frags",
        content=mbuf.tobytes(),
        compress=None,
        content_type="application/x-mapbuffer",
        cache_control=False,
    )

    # "list[dict[str, int]]" is not
    # "Sequence[MutableMapping[str, bool | int | float | str | list[bool | int | float | str]]]"
    seg_ids = [str(seg_id) for seg_id in skeletons.keys()]
    bounding_boxes = [Bbox.from_points(skel.vertices).to_list() for skel in skeletons.values()]
    datas = [
        {
            "x_start": int(bbox[0]),
            "y_start": int(bbox[1]),
            "z_start": int(bbox[2]),
            "x_stop": int(bbox[3]),
            "y_stop": int(bbox[4]),
            "z_stop": int(bbox[5]),
        }
        for bbox in bounding_boxes
    ]
    seg_db[
        seg_ids,
        ("x_start", "y_start", "z_start", "x_stop", "y_stop", "z_stop"),
    ] = datas  # type: ignore
    uuids = [str(uuid.uuid5(uuid.NAMESPACE_OID, idx.pformat() + seg_id)) for seg_id in seg_ids]
    datas_seg_id_frag = [
        {
            "seg_id": seg_id,
            "frag_fn": f"{Bbox(idx.start, idx.stop).to_filename()}.frags",
        }
        for seg_id in seg_ids
    ]
    frag_db[
        uuids,
        ("seg_id", "frag_fn"),
    ] = datas_seg_id_frag  # type: ignore
    # "list[dict[str, str]]" is not
    # "Sequence[MutableMapping[str, bool | int | float | str | list[bool | int | float | str]]]"
