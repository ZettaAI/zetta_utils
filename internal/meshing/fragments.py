from __future__ import annotations

import re
import uuid
from typing import Any, Sequence

import attrs
import DracoPy
import fastremap
import numpy as np
import zmesh
from cloudfiles import CloudFiles
from cloudvolume import Mesh
from cloudvolume.lib import Bbox
from mapbuffer import MapBuffer

from zetta_utils import builder, log, mazepa
from zetta_utils.geometry import Vec3D

# from . import mesh_graphene_remap
from zetta_utils.layer.db_layer import DBLayer
from zetta_utils.layer.volumetric import VolumetricIndex, VolumetricLayer
from zetta_utils.mazepa import semaphore

logger = log.get_logger("zetta_utils")

## Most of this file is a direct port of the
## sharded mesh generation in igneous to zetta_utils:
## https://github.com/seung-lab/igneous


def get_draco_encoding_settings(
    shape: Vec3D,
    offset: Vec3D,
    resolution: Vec3D,
    compression_level: int,
    create_metadata: bool,
    uses_new_draco_bin_size=False,
) -> dict[str, Any]:
    """
    Computes the Draco encoding settings.
    """
    chunk_offset_nm = offset * resolution

    min_quantization_range = max(shape * resolution)
    if uses_new_draco_bin_size:
        max_draco_bin_size = np.floor(min(resolution) / 2)
    else:
        max_draco_bin_size = np.floor(min(resolution) / np.sqrt(2))

    (
        draco_quantization_bits,
        draco_quantization_range,
        draco_bin_size,
    ) = calculate_draco_quantization_bits_and_range(min_quantization_range, max_draco_bin_size)
    draco_quantization_origin = chunk_offset_nm - (chunk_offset_nm % draco_bin_size)
    return {
        "quantization_bits": draco_quantization_bits,
        "compression_level": compression_level,
        "quantization_range": draco_quantization_range,
        "quantization_origin": draco_quantization_origin,
        "create_metadata": create_metadata,
    }


def calculate_draco_quantization_bits_and_range(
    min_quantization_range: int, max_draco_bin_size: int, draco_quantization_bits=None
) -> tuple[int, int, int]:
    """
    Computes draco parameters for integer quantizing the meshes.
    """
    if draco_quantization_bits is None:
        draco_quantization_bits = np.ceil(np.log2(min_quantization_range / max_draco_bin_size + 1))
    num_draco_bins = 2 ** draco_quantization_bits - 1
    draco_bin_size = np.ceil(min_quantization_range / num_draco_bins)
    draco_quantization_range = draco_bin_size * num_draco_bins
    if draco_quantization_range < min_quantization_range + draco_bin_size:
        if draco_bin_size == max_draco_bin_size:
            return calculate_draco_quantization_bits_and_range(
                min_quantization_range, max_draco_bin_size, draco_quantization_bits + 1
            )
        else:
            draco_bin_size = draco_bin_size + 1
            draco_quantization_range = draco_quantization_range + num_draco_bins
    return draco_quantization_bits, draco_quantization_range, draco_bin_size


@builder.register("MakeMeshFragsOperation")
@mazepa.taskable_operation_cls
@attrs.frozen()
class MakeMeshFragsOperation:  # pylint: disable=no-self-use
    crop_pad: Sequence[int] = (0, 0, 0)

    def get_operation_name(self):
        return "MakeMeshFragsOperation"

    def get_input_resolution(self, dst_resolution: Vec3D[float]) -> Vec3D[float]:
        return dst_resolution  # TODO add support for data res

    def with_added_crop_pad(self, crop_pad: Vec3D[int]) -> MakeMeshFragsOperation:
        return attrs.evolve(self, crop_pad=Vec3D(*self.crop_pad) + crop_pad)

    def __call__(  # pylint: disable=too-many-locals
        self,
        idx: VolumetricIndex,
        dst: VolumetricLayer,
        segmentation: VolumetricLayer,
        seg_db: DBLayer,
        frag_db: DBLayer,
        mesh_dir: str,
        frag_path: str | None = None,
        high_padding: int = 1,
        low_padding: int = 0,
        simplification_factor: int = 100,
        max_simplification_error: int = 40,
        draco_compression_level: int = 1,
        draco_create_metadata: bool = False,
        object_ids: list | None = None,
        closed_dataset_edges: bool = True,
        num_splits: Sequence[int] = (1, 1, 1),
    ):
        """
        Makes a Draco compressed mesh fragment file for a given index, and records all
        segments in the index to a database for future sharding. Note the following
        five differences to the igneous implementation:

        1) Only fragments and not individual meshes are supported.
        2) Fragments contain Draco compressed meshes, not raw.
        3) Instead of spatial index, two DBLayers are used to keep track of metadata.
        4) Filling missing areas with zeros should be set in VolumetricLayer.
        5) Splitting the index for cutting down on number of fragments is supported.

        :param idx: Index to turn into a fragment.
        :param segmentation: The segmentation to use.
        :param dst: Duplicate argument for ``segmentation`` - only exists
            to let subchunkable access the segmentation for generating bounds.
        :param seg_db: The DBLayer that contains each segment id - only used to keep
            track of the number of segments (necessary for sharding parameter computation).
        :param frag_db: The DBLayer that contains each segment id and fragment name, for
            all segids in each fragment. This database is used to look up fragments when
            sharding a given segid during sharding.
        :param mesh_dir: The mesh directory for the run; the fragments will be in
            `{frag_path}/{mesh_dir}`, and the shards will be in `{segmentation_path}/{mesh_dir}`.
            Note that this operation does NOT modify the infofile of the segmentation with the
            `mesh_dir` information - this is handled during sharding.
            MUST START WITH `mesh_mip_[0-9]+` FOR SHARDING TO DETECT MIP, WHICH IS RELATIVE TO
            THE SEGMENTATION INFOFILE.
        :param frag_path: Absolute path for where to store the fragments (with the `mesh_dir`
            prefix on the filenames). If not provided, defaults to the segmentation folder.
        :param high_padding: Padding on the high side; recommended to keep default.
        :param low_padding: Padding on the low side; recommended to keep default.
        :param simplification factor: What factor to try to reduce the number of triangles
            in the mesh, constrained by `max_simplification_error`.
        :param max_simplification_error: The maximum physical distance that
            simplification is allowed to move a triangle vertex by.
        :param draco_compression_level: Draco compression level.
        :param draco_create_metadata: Whether to create Draco metadata.
        :param object_ids: If provided, only generate fragments for these ids.
        :param closed_dataset_edges: Close the mesh faces at the edge of the dataset.
        :param num_splits: Split the `idx` into `(NUM_X, NUM_Y, NUM_Z)` chunks when
            meshing. zmesh uses double the memory if the size of an object exceeds
            (1023, 1023, 511) in any axis due to the 32-bit limitation, so dividing
            the idx and processing in chunks keeps the size of the objects below
            this limit and reduces the number of fragments.
        """

        if mesh_dir is not None and not re.match(r"^mesh_mip_[0-9]+", mesh_dir):
            raise ValueError(
                "``mesh_dir`` MUST start with `mesh_mip_[0-9]+` for sharding "
                "to work; this is because the mesh MIP is detected from the "
                f"mesh directory name. Received {mesh_dir}"
            )

        # Marching cube needs 1vx overlaps to not have lines appear
        # between adjacent chunks.
        idx_padded = idx.translated_start(
            (-low_padding, -low_padding, -low_padding)
        ).translated_end((high_padding, high_padding, high_padding))

        idx_padded_full = idx_padded.intersection(segmentation.backend.get_bounds(idx.resolution))

        draco_encoding_settings = get_draco_encoding_settings(
            shape=idx_padded_full.shape,
            offset=idx_padded_full.start,
            resolution=idx_padded_full.resolution,
            compression_level=draco_compression_level,
            create_metadata=draco_create_metadata,
            uses_new_draco_bin_size=False,
        )

        subidxs = [
            subidx
            for split_idx in idx.split(num_splits)
            if (
                subidx := split_idx.intersection(segmentation.backend.get_bounds(idx.resolution))
                .translated_start((-low_padding, -low_padding, -low_padding))
                .translated_end((high_padding, high_padding, high_padding))
            ).get_size()
            > 0
        ]

        # TODO agglomerate, timestamp, stop_layer for graphene
        with semaphore("read"):
            data = segmentation[idx_padded_full][0]

        if not np.any(data):
            return

        if object_ids:
            data = fastremap.mask_except(data, object_ids, in_place=True)

        data, renumbermap = fastremap.renumber(data, in_place=True)
        renumbermap = {v: k for k, v in renumbermap.items()}
        meshers = [zmesh.Mesher(subidx.resolution) for subidx in subidxs]

        left_offsets = [Vec3D(0, 0, 0) for idx in subidxs]
        for i, subidx in enumerate(subidxs):
            data_sub = data[subidx.get_intersection_and_subindex(idx_padded_full)[1]]
            if closed_dataset_edges:
                data_sub, left_offset = _handle_dataset_boundary(data_sub, segmentation, subidx)
                left_offsets[i] = left_offset
                meshers[i].mesh(data_sub.T)
            del data_sub
        del data

        _compute_meshes(
            idx,
            segmentation,
            subidxs,
            meshers,
            renumbermap,
            left_offsets,
            mesh_dir=mesh_dir,
            frag_path=frag_path,
            seg_db=seg_db,
            frag_db=frag_db,
            simplification_factor=simplification_factor,
            max_simplification_error=max_simplification_error,
            draco_encoding_settings=draco_encoding_settings,
        )


def _handle_dataset_boundary(
    data: np.ndarray, segmentation: VolumetricLayer, idx: VolumetricIndex
):
    """
    This logic is used to add a black border along sides
    of the image that touch the dataset boundary which
    results in the closure of the mesh faces on that side.
    """
    dataset_bounds = segmentation.backend.get_bounds(idx.resolution)
    if not any(idx.aligned(dataset_bounds)):
        return data, Vec3D(0, 0, 0)

    shape = [*data.shape]
    offset = [0, 0, 0]
    for i in range(3):
        if idx.start[i] == dataset_bounds.start[i]:
            offset[i] += 1
            shape[i] += 1
        if idx.stop[i] == dataset_bounds.stop[i]:
            shape[i] += 1

    slices = (
        slice(offset[0], offset[0] + data.shape[0]),
        slice(offset[1], offset[1] + data.shape[1]),
        slice(offset[2], offset[2] + data.shape[2]),
    )

    mirror_data = np.zeros(shape, dtype=data.dtype, order="F")
    mirror_data[slices] = data
    if offset[0]:
        mirror_data[0, :, :] = 0
    if offset[1]:
        mirror_data[:, 0, :] = 0
    if offset[2]:
        mirror_data[:, :, 0] = 0

    return mirror_data, Vec3D(*offset)


def _compute_meshes(
    idx: VolumetricIndex,
    segmentation: VolumetricLayer,
    subidxs: list[VolumetricIndex],
    meshers: list[zmesh.Mesher],
    renumbermap: dict[int, int],
    left_bound_offsets: list[Vec3D],
    mesh_dir: str,
    frag_path: str | None,
    seg_db: DBLayer,
    frag_db: DBLayer,
    simplification_factor: int,
    max_simplification_error: int,
    draco_encoding_settings: dict[str, Any],
):  # pylint: disable = too-many-locals
    bounding_boxes = {}
    meshes = {}
    for obj_id in set().union(*(set(mesher.ids()) for mesher in meshers)):
        remapped_id = renumbermap[obj_id]
        mesh_binary, mesh_bounds = _combine_and_create_mesh(
            subidxs,
            meshers,
            obj_id,
            left_bound_offsets,
            simplification_factor=simplification_factor,
            max_simplification_error=max_simplification_error,
            draco_encoding_settings=draco_encoding_settings,
        )
        bounding_boxes[remapped_id] = mesh_bounds.to_list()
        meshes[remapped_id] = mesh_binary

    frag_path_to_use = frag_path or segmentation.name
    cf = CloudFiles(frag_path_to_use)

    mbuf = MapBuffer(meshes, compress="br")
    cf.put(
        f"{mesh_dir}/{Bbox(idx.start, idx.stop).to_filename()}.frags",
        content=mbuf.tobytes(),
        compress=None,
        content_type="application/x.mapbuffer",
        cache_control=False,
    )
    seg_ids = [str(seg_id) for seg_id in bounding_boxes]
    datas = [{"exists": True} for _ in seg_ids]
    seg_db[
        seg_ids,
        ("exists",),
    ] = datas  # type: ignore
    # "list[dict[str, int]]" is not
    # "Sequence[MutableMapping[str, bool | int | float | str | list[bool | int | float | str]]]"

    uuids = [str(uuid.uuid5(uuid.NAMESPACE_OID, idx.pformat() + seg_id)) for seg_id in seg_ids]
    datas_seg_id_frag = [
        {
            "seg_id": str(seg_id),
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


def _combine_and_create_mesh(
    subidxs: list[VolumetricIndex],
    meshers: list[zmesh.Mesher],
    obj_id: int,
    left_bound_offsets: list[Vec3D],
    simplification_factor: int,
    max_simplification_error: int,
    draco_encoding_settings: dict[str, Any],
):
    meshes = [
        mesher.get_mesh(
            obj_id,
            simplification_factor=simplification_factor,
            max_simplification_error=max_simplification_error,
            voxel_centered=True,
        )
        for mesher in meshers
    ]

    for mesher in meshers:
        mesher.erase(obj_id)

    for mesh, subidx, left_bound_offset in zip(meshes, subidxs, left_bound_offsets):
        mesh.vertices[:] += np.array((subidx.start - left_bound_offset) * subidx.resolution)
        # zmesh does not initialise these attributes even though we need them
        mesh.normals = np.array([], dtype=np.float32).reshape((0, 3))
        mesh.encoding_type = "draco"

    combined_mesh = Mesh.concatenate(*meshes).consolidate()

    # TODO: Add option to run another round of simplification

    for mesh in meshes:
        del mesh
    del meshes
    combined_mesh_bounds = Bbox(
        np.amin(combined_mesh.vertices, axis=0), np.amax(combined_mesh.vertices, axis=0)
    )

    combined_mesh_binary = DracoPy.encode(  # pylint: disable = c-extension-no-member
        combined_mesh.vertices, combined_mesh.faces, **draco_encoding_settings
    )

    del combined_mesh

    return combined_mesh_binary, combined_mesh_bounds
