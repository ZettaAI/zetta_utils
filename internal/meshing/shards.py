# pylint: disable = c-extension-no-member
from __future__ import annotations

import copy
import functools
import math
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from time import strftime
from typing import Tuple

import DracoPy
import numpy as np
import pyfqmr
import shardcomputer
import trimesh
from cloudfiles import CloudFile, CloudFiles, paths
from cloudvolume import CloudVolume, Mesh
from cloudvolume.datasource.precomputed.mesh.multilod import (
    MultiLevelPrecomputedMeshManifest,
    to_stored_model_space,
)
from cloudvolume.datasource.precomputed.sharding import (
    ShardingSpecification,
    synthesize_shard_files,
)
from cloudvolume.lib import Vec, first, sip
from mapbuffer import MapBuffer
from tqdm import tqdm

from zetta_utils import builder, log, mazepa
from zetta_utils.layer.db_layer import DBLayer

## Most of this file is a direct port of the
## sharded mesh generation in igneous to zetta_utils:
## https://github.com/seung-lab/igneous

logger = log.get_logger("zetta_utils")


@builder.register("build_make_mesh_shards_flow")
def build_make_mesh_shards_flow(
    segmentation_path: str,
    seg_db: DBLayer,
    frag_db: DBLayer,
    shard_index_bytes: int = 2 ** 13,
    minishard_index_bytes: int = 2 ** 15,
    min_shards: int = 1,
    num_frags_per_shard_no_task: int = 1,
    num_lods: int = 0,
    draco_compression_level: int = 7,
    vertex_quantization_bits: int = 16,
    minishard_index_encoding: str = "gzip",
    mesh_dir: str | None = None,
    frag_path: str | None = None,
    min_chunk_size: Tuple[int, int, int] = (256, 256, 256),
    max_seg_ids_per_shard: int | None = None,
) -> mazepa.Flow:
    """
    Wrapper that makes a Flow to make shards; see MakeMeshShardsFlowSchema.
    """
    return MakeMeshShardsFlowSchema()(
        segmentation_path=segmentation_path,
        seg_db=seg_db,
        frag_db=frag_db,
        shard_index_bytes=shard_index_bytes,
        minishard_index_bytes=minishard_index_bytes,
        min_shards=min_shards,
        num_frags_per_shard_no_task=num_frags_per_shard_no_task,
        num_lods=num_lods,
        draco_compression_level=draco_compression_level,
        vertex_quantization_bits=vertex_quantization_bits,
        minishard_index_encoding=minishard_index_encoding,
        mesh_dir=mesh_dir,
        frag_path=frag_path,
        min_chunk_size=min_chunk_size,
        max_seg_ids_per_shard=max_seg_ids_per_shard,
    )


@builder.register("MakeShardsFlowSchema")
@mazepa.flow_schema_cls
class MakeMeshShardsFlowSchema:
    def flow(  # pylint: disable=too-many-locals, no-self-use
        self,
        segmentation_path: str,
        seg_db: DBLayer,
        frag_db: DBLayer,
        shard_index_bytes: int,
        minishard_index_bytes: int,
        min_shards: int,
        num_frags_per_shard_no_task: int,
        num_lods: int,
        draco_compression_level: int,
        vertex_quantization_bits: int,
        minishard_index_encoding: str,
        mesh_dir: str | None,
        frag_path: str | None,
        min_chunk_size: Tuple[int, int, int],
        max_seg_ids_per_shard: int | None,
        worker_type: str | None = None,
    ) -> mazepa.FlowFnReturnType:
        """
        Combines meshes from multiple fragments and stores it into shards at multiple
        resolutions.
        Wrapped as a FlowSchema and a build function to avoid the mazepa id generation
        issue; please see the build function for default values.

        :param segmentation_path: The segmentation path.
        :param seg_db: The DBLayer that contains each segment id - only used to keep
            track of the number of segments (necessary for sharding parameter computation).
        :param frag_db: The DBLayer that contains each segment id and fragment name, for
            all segids in each fragment. The flow will generate tasks to iterate through
            this database and assign each segid to a shard, and then will reference the
            assigned shard number to figure out which segids belong to the shard.
        :param mesh_dir: The mesh directory for the run; the fragments will be in
            `{frag_path}/{mesh_dir}`, and the shards will be in `{segmentation_path}/{mesh_dir}`.
            Note that if the segmentation infofile already has the mesh directory information,
            this parameter _must_ either not be provided or agree with the existing information.
            MUST START WITH `mesh_mip_[0-9]+` FOR SHARDING TO DETECT MIP, WHICH IS RELATIVE TO THE
            SEGMENTATION INFOFILE.
        :param frag_path: Where the fragments are stored (with the `mesh_dir` prefix on
            the filenames). If not provided, defaults to the segmentation folder.
        :param shard_index_bytes: Sharding parameter; see `_compute_shard_params_for_hashed`.
        :param minishard_index_bytes: Sharding parameter; see `_compute_shard_params_for_hashed`.
        :param min_shards: Sharding parameter; see `_compute_shard_params_for_hashed`.
        :param num_frags_per_shard_no_task: Number of tasks to generate for assigning shard numbers
            to mesh fragments in the `frag_db`.
        :param num_lods: Number of LODs (mesh equivalent of MIPs) to generate.
        :param draco_compression_level: Draco compression level.
        :param minishard_index_encoding: Minishard index encoding; see
            "https://github.com/seung-lab/igneous/blob/
            7258212e57c5cfc1e5f0de8162f830c7d49e1be9/igneous/task_creation/image.py#L343"
            for details.
        :param max_seg_ids_per_shard: Max number of segment ids that can be assigned
            to a single shard.
        :param min_chunk_size: Minimum chunk size for a mesh to be split during
            LOD generation.
        """

        if mesh_dir is not None and not re.match(r"^mesh_mip_[0-9]+", mesh_dir):
            raise ValueError(
                "`mesh_dir` MUST start with `mesh_mip_[0-9]+` for sharding "
                "to work; this is because the mesh MIP is detected from the "
                f"mesh directory name. Received {mesh_dir}"
            )

        mesh_info, mesh_dir_to_use = _configure_multires_info(
            segmentation_path, vertex_quantization_bits, mesh_dir
        )

        num_seg_ids = len(seg_db)

        if max_seg_ids_per_shard is not None:
            assert max_seg_ids_per_shard >= 1
            min_shards = max(int(np.ceil(num_seg_ids / max_seg_ids_per_shard)), min_shards)

        (shard_bits, minishard_bits, preshift_bits) = _compute_shard_params_for_hashed(
            num_seg_ids=num_seg_ids,
            shard_index_bytes=int(shard_index_bytes),
            minishard_index_bytes=int(minishard_index_bytes),
            min_shards=min_shards,
        )

        spec = ShardingSpecification(
            type="neuroglancer_uint64_sharded_v1",
            preshift_bits=preshift_bits,
            hash="murmurhash3_x86_128",
            minishard_bits=minishard_bits,
            shard_bits=shard_bits,
            minishard_index_encoding=minishard_index_encoding,
            data_encoding="raw",  # draco encoded meshes
        )

        cv = CloudVolume(segmentation_path)
        cv.mesh.meta.info = mesh_info  # ensure no race conditions
        cv.mesh.meta.info["sharding"] = spec.to_dict()
        cv.mesh.meta.commit_info()

        # rebuild b/c sharding changes the mesh source class
        cv = CloudVolume(segmentation_path, progress=True)
        cv.mip = cv.mesh.meta.mip

        # assign shard numbers distributedly
        num_batches = math.ceil(len(frag_db) / num_frags_per_shard_no_task)
        shard_no_tasks = [
            _compute_shard_no_for_fragments.make_task(
                frag_db,
                batch_num,
                num_frags_per_shard_no_task,
                preshift_bits,
                shard_bits,
                minishard_bits,
            ).with_worker_type(worker_type)
            for batch_num in range(num_batches)
        ]
        yield shard_no_tasks
        yield mazepa.Dependency()
        for task in shard_no_tasks:
            # dryrun handling
            if task.outcome is None:
                return
            assert task.outcome.return_value is not None
        shard_numbers = set().union(*(task.outcome.return_value for task in shard_no_tasks))  # type: ignore # pylint: disable=line-too-long
        cv.provenance.processing.append(
            {
                "method": {
                    "task": "MultiResShardedMeshMergeTask",
                    "segmentation_path": segmentation_path,
                    "mip": cv.mesh.meta.mip,
                    "num_lods": num_lods,
                    "vertex_quantization_bits": vertex_quantization_bits,
                    "preshift_bits": preshift_bits,
                    "minishard_bits": minishard_bits,
                    "shard_bits": shard_bits,
                    "mesh_dir": mesh_dir_to_use,
                    "frag_path": frag_path,
                    "draco_compression_level": draco_compression_level,
                    "min_chunk_size": min_chunk_size,
                },
                "by": "zetta_utils",
                "date": strftime("%Y-%m-%d %H:%M %Z"),
            }
        )
        cv.commit_provenance()

        tasks = [
            _multires_sharded_mesh_merge.make_task(
                segmentation_path,
                frag_db,
                shard_no,
                num_lods=num_lods,
                mesh_dir=mesh_dir_to_use,
                frag_path=frag_path,
                draco_compression_level=draco_compression_level,
                min_chunk_size=min_chunk_size,
            ).with_worker_type(worker_type)
            for shard_no in shard_numbers
        ]

        yield tasks
        yield mazepa.Dependency()
        for task in tasks:
            assert task.outcome is not None


def _configure_multires_info(
    segmentation_path: str, vertex_quantization_bits: int, mesh_dir: str | None
):
    """
    Computes properties and uploads a multires mesh info file.
    """
    assert vertex_quantization_bits in (10, 16), vertex_quantization_bits

    vol = CloudVolume(segmentation_path)

    mesh_dir = mesh_dir or vol.info.get("mesh", None)
    if mesh_dir is None:
        raise ValueError(
            f"`mesh_dir` has not been specified, but the infofile at {segmentation_path} "
            "does not specify the mesh directory."
        )

    if not "mesh" in vol.info:
        vol.info["mesh"] = mesh_dir
        vol.commit_info()
    else:
        if not vol.info["mesh"] == mesh_dir:
            raise ValueError(
                f"`mesh_dir` has been specified as {mesh_dir}, but the infofile at "
                f"{segmentation_path} specifies a different directory, {vol.info['mesh']}"
            )

    if vol.mesh.meta.mip is None:
        raise ValueError(
            "Unable to detect mesh resolution. "
            "Please specify the resolution in the mesh info file."
        )

    res = vol.meta.resolution(vol.mesh.meta.mip)
    cf = CloudFiles(segmentation_path)
    info_filename = f"{mesh_dir}/info"
    mesh_info = cf.get_json(info_filename) or {}
    new_mesh_info = copy.deepcopy(mesh_info)
    new_mesh_info["@type"] = "neuroglancer_multilod_draco"
    new_mesh_info["vertex_quantization_bits"] = vertex_quantization_bits
    new_mesh_info["transform"] = [
        res[0],
        0,
        0,
        0,
        0,
        res[1],
        0,
        0,
        0,
        0,
        res[2],
        0,
    ]
    new_mesh_info["lod_scale_multiplier"] = 1.0

    if new_mesh_info != mesh_info:
        cf.put_json(info_filename, new_mesh_info, cache_control="no-cache")

    return new_mesh_info, mesh_dir


# TODO: Refactor duplicate with Skeleton Sharding, alongside where this is called
def _compute_shard_params_for_hashed(
    num_seg_ids: int,
    shard_index_bytes: int = 2 ** 13,
    minishard_index_bytes: int = 2 ** 15,
    min_shards: int = 1,
):
    """
    Computes the shard parameters for objects that
    have been randomly hashed (e.g. murmurhash) so
    that the keys are evenly distributed. This is
    applicable to skeletons and meshes.

    The equations come from the following assumptions.
    a. The keys are approximately uniformly randomly distributed.
    b. Preshift bits aren't useful for random keys so are zero.
    c. Our goal is to optimize the size of the shard index and
        the minishard indices to be reasonably sized. The default
        values are set for a 100 Mbps connection.
    d. The equations below come from finding a solution to
        these equations given the constraints provided.

     num_shards * num_minishards_per_shard
                = 2^(shard_bits) * 2^(minishard_bits)
                = num_seg_ids_in_dataset / seg_ids_per_minishard

            # from defininition of minishard_bits assuming fixed capacity
            seg_ids_per_minishard = minishard_index_bytes / 3 / 8

            # from definition of minishard bits
            minishard_bits = ceil(log2(shard_index_bytes / 2 / 8))

    Returns: (shard_bits, minishard_bits, preshift_bits)
    """
    assert min_shards >= 1
    if num_seg_ids <= 0:
        return (0, 0, 0)

    num_minishards_per_shard = shard_index_bytes / 2 / 8
    seg_ids_per_minishard = minishard_index_bytes / 3 / 8
    seg_ids_per_shard = num_minishards_per_shard * seg_ids_per_minishard

    if num_seg_ids >= seg_ids_per_shard:
        minishard_bits = np.ceil(np.log2(num_minishards_per_shard))
        shard_bits = np.ceil(
            np.log2(num_seg_ids / (seg_ids_per_minishard * (2 ** minishard_bits)))
        )
    elif num_seg_ids >= seg_ids_per_minishard:
        minishard_bits = np.ceil(np.log2(num_seg_ids / seg_ids_per_minishard))
        shard_bits = 0
    else:
        minishard_bits = 0
        shard_bits = 0

    capacity = seg_ids_per_shard * (2 ** shard_bits)
    utilized_capacity = num_seg_ids / capacity

    # Try to pack shards to capacity, allow going
    # about 10% over the input level.
    if utilized_capacity <= 0.55:
        shard_bits -= 1

    shard_bits = max(shard_bits, 0)
    min_shard_bits = np.round(np.log2(min_shards))

    delta = max(min_shard_bits - shard_bits, 0)
    shard_bits += delta
    minishard_bits -= delta

    shard_bits = max(shard_bits, min_shard_bits)
    minishard_bits = max(minishard_bits, 0)

    return (int(shard_bits), int(minishard_bits), 0)


# TODO: Refactor duplicate with Skeleton Sharding, alongside where this is called
@mazepa.taskable_operation
def _compute_shard_no_for_fragments(
    frag_db: DBLayer,
    batch_num: int,
    batch_size: int,
    preshift_bits: int,
    shard_bits: int,
    minishard_bits: int,
) -> list[str]:
    """
    Computes and updates the shard number for a batch of fragments in the
    fragments database based on the seg id.

    :param frag_db: Fragments database containing all mesh fragments.
    :param batch_num: Which batch to process.
    :param batch_size: The size of each batch.
    :param preshift_bits: Sharding parameter; see `_compute_shard_params_for_hashed`.
    :param shard_bits: Sharding parameter; see `_compute_shard_params_for_hashed`.
    :param minishard_bits: Sharding parameter; see `_compute_shard_params_for_hashed`.
    """
    seg_id_and_frags = frag_db.get_batch(batch_num, batch_size, return_columns=("seg_id",))
    if len(seg_id_and_frags) == 0:
        return []
    shard_numbers = [
        shardcomputer.shard_number(int(res["seg_id"]), preshift_bits, shard_bits, minishard_bits)
        for res in seg_id_and_frags.values()
    ]
    frag_db[seg_id_and_frags.keys(), ("shard_no")] = [
        {"shard_no": shard_no} for shard_no in shard_numbers
    ]
    return shard_numbers


@mazepa.taskable_operation
def _multires_sharded_mesh_merge(
    segmentation_path: str,
    frag_db: DBLayer,
    shard_no: str,
    draco_compression_level: int,
    num_lods: int,
    min_chunk_size: Tuple[int, int, int],
    mesh_dir: str,
    frag_path: str | None,
    progress: bool = False,
):
    """
    Creates and uploads the multires shard of the given shard number
    by merging mesh fragments for seg ids in the shard and creating the LODs required.

    :param segmentation_path: The segmentation path.
    :param frag_db: The DBLayer that contains each segment id and fragment name, as
    well as the shard number for all segids in each fragment.
    :param shard_no: The shard number to generate.
    :param draco_compression_level: Draco compression level.
    :param mesh_dir: The mesh directory for the run; the fragments should be in
        `{frag_path}/{mesh_dir}`, and the shards will be in `{segmentation_path}/{mesh_dir}`.
    :param frag_path: Where the fragments are stored (with the `mesh_dir` prefix on
            the filenames). If not provided, defaults to the segmentation folder.
    :param num_lods: Number of LODs (mesh equivalent of MIPs) to generate.
    :param min_chunk_size: Minimum chunk size for a mesh to be split during
        LOD generation.
    :param progress: Whether to show a progress bar.
    """
    cv = CloudVolume(segmentation_path)
    cv.mip = cv.mesh.meta.mip

    query_results = frag_db.query({"shard_no": [shard_no]}, return_columns=("seg_id", "frag_fn"))
    # TODO: return type is a little too broad?
    seg_ids = list(set(int(query_result["seg_id"]) for query_result in query_results.values()))  # type: ignore # pylint: disable=line-too-long
    filenames = list(set(str(query_result["frag_fn"]) for query_result in query_results.values()))

    meshes = _collect_mesh_fragments(cv, seg_ids, filenames, mesh_dir, frag_path, progress)
    del filenames

    # important to iterate this way to avoid
    # creating a copy of meshes vs. { ... for in }
    for seg_id in seg_ids:
        meshes[seg_id] = Mesh.concatenate(*meshes[seg_id]).consolidate()
    del seg_ids

    fname, shard = _create_mesh_shard(
        cv, meshes, num_lods, draco_compression_level, progress, shard_no, min_chunk_size
    )
    del meshes

    if shard is None:
        return

    cf = CloudFiles(cv.mesh.meta.layerpath)
    cf.put(
        fname,
        shard,
        compress=False,
        content_type="application/octet-stream",
        cache_control="no-cache",
    )


def _create_mesh_shard(
    cv: CloudVolume,
    meshes: dict[int, Mesh],
    num_lods: int,
    draco_compression_level: int,
    progress: bool,
    shard_no: str,
    min_chunk_size: Tuple[int, int, int],
):
    meshes = {
        seg_id: _process_mesh(
            cv,
            seg_id,
            mesh,
            num_lods,
            min_chunk_size,
            draco_compression_level=draco_compression_level,
        )
        for seg_id, mesh in tqdm(meshes.items(), disable=not progress)
    }
    data_offset = {
        seg_id: len(manifest)
        for seg_id, (manifest, mesh_binary) in meshes.items()
        if manifest is not None and len(mesh_binary) > 0
    }
    meshes = {
        seg_id: mesh_binary + manifest.to_binary()
        for seg_id, (manifest, mesh_binary) in meshes.items()
        if manifest is not None and len(mesh_binary) > 0
    }

    if len(meshes) == 0:
        return None, None

    shard_files = synthesize_shard_files(cv.mesh.reader.spec, meshes, data_offset)

    if len(shard_files) != 1:
        raise ValueError(
            "Only one shard file should be generated per task. "
            f"Expected: {shard_no}, Got: {shard_files.keys()}"
        )

    filename = first(shard_files.keys())
    return filename, shard_files[filename]


def _process_mesh(
    cv: CloudVolume,
    seg_id: int,
    mesh: Mesh,
    num_lods: int,
    min_chunk_size: Tuple[int, int, int] = (512, 512, 512),
    draco_compression_level: int = 7,
) -> Tuple[MultiLevelPrecomputedMeshManifest, Mesh]:

    mesh.vertices /= cv.meta.resolution(cv.mesh.meta.mip)

    grid_origin = np.floor(np.min(mesh.vertices, axis=0))
    mesh_shape = (np.max(mesh.vertices, axis=0) - grid_origin).astype(int)

    if np.any(mesh_shape == 0):
        return (None, None)

    min_chunk_size_np = np.array(min_chunk_size, dtype=int)
    max_lod = int(max(np.min(np.log2(mesh_shape / min_chunk_size_np)), 0))
    max_lod = min(max_lod, num_lods)

    lods = _generate_lods(mesh, max_lod)
    chunk_shape = np.ceil(mesh_shape / (2 ** (len(lods) - 1)))

    if np.any(chunk_shape == 0):
        return (None, None)

    lods = [
        _create_octree_level_from_mesh(lods[lod], chunk_shape, lod, len(lods))
        for lod in range(len(lods))
    ]
    fragment_positions = [nodes for submeshes, nodes in lods]
    lods = [submeshes for submeshes, nodes in lods]

    manifest = MultiLevelPrecomputedMeshManifest(
        segment_id=seg_id,
        chunk_shape=chunk_shape,
        grid_origin=grid_origin,
        num_lods=len(lods),
        lod_scales=[2 ** i for i in range(len(lods))],
        vertex_offsets=[[0, 0, 0]] * len(lods),
        num_fragments_per_lod=[len(lods[lod]) for lod in range(len(lods))],
        fragment_positions=fragment_positions,
        fragment_offsets=[],  # needs to be set when we have the final value
    )

    vqb = int(cv.mesh.meta.info["vertex_quantization_bits"])

    mesh_binaries = []
    for lod, submeshes in enumerate(lods):
        for frag_no, submesh in enumerate(submeshes):
            submesh.vertices = to_stored_model_space(
                submesh.vertices,
                manifest,
                lod=lod,
                vertex_quantization_bits=vqb,
                frag=frag_no,
            )

            minpt = np.min(submesh.vertices, axis=0)
            quantization_range = np.max(submesh.vertices, axis=0) - minpt
            quantization_range = np.max(quantization_range)

            # mesh.vertices must be integer type or mesh will display
            # distorted in neuroglancer.
            try:
                submesh = DracoPy.encode(
                    submesh.vertices,
                    submesh.faces,
                    quantization_bits=vqb,
                    compression_level=draco_compression_level,
                    quantization_range=quantization_range,
                    quantization_origin=minpt,
                    create_metadata=True,
                )
            except DracoPy.EncodingFailedException:
                submesh = b""

            manifest.fragment_offsets.append(len(submesh))
            mesh_binaries.append(submesh)

    return (manifest, b"".join(mesh_binaries))


def _collect_mesh_fragments(
    cv: CloudVolume,
    seg_ids: list[int],
    filenames: list[str],
    mesh_dir: str,
    frag_path: str | None,
    progress: bool = False,
) -> dict[int, list[Mesh]]:
    filenames = [cv.meta.join(mesh_dir, loc) for loc in filenames]

    block_size = 20

    if len(filenames) < block_size:
        blocks = [filenames]
        n_blocks = 1
    else:
        n_blocks = max(len(filenames) // block_size, 1)
        blocks = sip(filenames, block_size)

    all_meshes = defaultdict(list)

    def process_shardfile(item):
        filename, content = item
        fragment = MapBuffer(content, frombytesfn=Mesh.from_draco)

        for seg_id in seg_ids:
            try:
                mesh = fragment[seg_id]
                mesh.id = seg_id
                all_meshes[seg_id].append((filename, mesh))
            except KeyError:
                continue

        if hasattr(content, "close"):
            content.close()

        return filename

    frag_prefix = frag_path or cv.cloudpath
    local_input = False
    if paths.extract(frag_prefix).protocol == "file":
        local_input = True
        frag_prefix = frag_prefix.replace("file://", "", 1)

    for filenames_block in tqdm(
        blocks, desc="Filename Block", total=n_blocks, disable=(not progress)
    ):
        if local_input:
            all_files = {}
            for filename in filenames_block:
                all_files[filename] = open(  # pylint: disable=consider-using-with
                    os.path.join(frag_prefix, filename), "rb"
                )

            for item in tqdm(all_files.items(), desc="Scanning Fragments", disable=not progress):
                process_shardfile(item)
        else:
            all_files = {
                filename: CloudFile(cv.meta.join(frag_prefix, filename), cache_meta=True)
                for filename in filenames_block
            }
            # TODO: Check for ThreadPoolExecutor compatibility with multithreading
            with ThreadPoolExecutor(max_workers=block_size) as executor:
                for filename in executor.map(process_shardfile, all_files.items()):
                    pass

    # ensure consistent results across multiple runs
    # by sorting mesh fragments by filename
    for seg_id in all_meshes:
        all_meshes[seg_id].sort(key=lambda pair: pair[0])
        all_meshes[seg_id] = [pair[1] for pair in all_meshes[seg_id]]

    return all_meshes


def _generate_lods(
    mesh: Mesh,
    num_lods: int,
    decimation_factor: int = 2,
    aggressiveness: float = 5.5,
):
    assert num_lods >= 0, num_lods

    lods = [mesh]

    # from pyfqmr documentation:
    # threshold = alpha * (iteration + K) ** agressiveness
    #
    # Threshold is the total error that can be tolerated by
    # deleting a vertex.
    for i in range(1, num_lods + 1):
        simplifier = pyfqmr.Simplify()
        simplifier.setMesh(mesh.vertices, mesh.faces)
        simplifier.simplify_mesh(
            target_count=max(int(len(mesh.faces) / (decimation_factor ** i)), 4),
            aggressiveness=aggressiveness,
            preserve_border=True,
            verbose=False,
            # Additional parameters to expose?
            # max_iterations=
            # K=
            # alpha=
            # update_rate=    # Number of iterations between each update.
            # lossless=
            # threshold_lossless=
        )

        lods.append(Mesh(*simplifier.getMesh()))

    return lods


## Below functions adapted from
## https://github.com/google/neuroglancer/issues/272


def _cmp_zorder(lhs, rhs) -> bool:
    def less_msb(x: int, y: int) -> bool:
        return x < y and x < (x ^ y)

    # Assume lhs and rhs array-like objects of indices.
    assert len(lhs) == len(rhs)
    # Will contain the most significant dimension.
    msd = 2
    # Loop over the other dimensions.
    for dim in [1, 0]:
        # Check if the current dimension is more significant
        # by comparing the most significant bits.
        if less_msb(lhs[msd] ^ rhs[msd], lhs[dim] ^ rhs[dim]):
            msd = dim
    return lhs[msd] - rhs[msd]


def _create_octree_level_from_mesh(mesh, chunk_shape, lod, num_lods):
    """
    Create submeshes by slicing the orignal mesh to produce smaller chunks
    by slicing them from x,y,z dimensions.

    This creates (2^lod)^3 submeshes.
    """
    if lod == num_lods - 1:
        return ([mesh], ((0, 0, 0),))

    mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)

    scale = Vec(*(np.array(chunk_shape) * (2 ** lod)))
    offset = Vec(*np.floor(mesh.vertices.min(axis=0)))
    grid_size = Vec(*np.ceil((mesh.vertices.max(axis=0) - offset) / scale), dtype=int)

    nx, ny, nz = np.eye(3)
    ox, oy, oz = offset * np.eye(3)

    submeshes = []
    nodes = []
    for x in range(0, grid_size.x):
        # list(...) required b/c it doesn't like Vec classes
        mesh_x = trimesh.intersections.slice_mesh_plane(
            mesh, plane_normal=nx, plane_origin=list(nx * x * scale.x + ox)
        )
        mesh_x = trimesh.intersections.slice_mesh_plane(
            mesh_x, plane_normal=-nx, plane_origin=list(nx * (x + 1) * scale.x + ox)
        )
        for y in range(0, grid_size.y):
            mesh_y = trimesh.intersections.slice_mesh_plane(
                mesh_x, plane_normal=ny, plane_origin=list(ny * y * scale.y + oy)
            )
            mesh_y = trimesh.intersections.slice_mesh_plane(
                mesh_y, plane_normal=-ny, plane_origin=list(ny * (y + 1) * scale.y + oy)
            )
            for z in range(0, grid_size.z):
                mesh_z = trimesh.intersections.slice_mesh_plane(
                    mesh_y, plane_normal=nz, plane_origin=list(nz * z * scale.z + oz)
                )
                mesh_z = trimesh.intersections.slice_mesh_plane(
                    mesh_z, plane_normal=-nz, plane_origin=list(nz * (z + 1) * scale.z + oz)
                )

                if len(mesh_z.vertices) == 0:
                    continue

                # test for totally degenerate meshes by checking if
                # all of two axes match, meaning the mesh must be a
                # point or a line.
                if (
                    np.sum(
                        [np.all(mesh_z.vertices[:, i] == mesh_z.vertices[0, i]) for i in range(3)]
                    )
                    >= 2
                ):
                    continue

                submeshes.append(mesh_z)
                nodes.append((x, y, z))

    # Sort in Z-curve order
    submeshes_ordered, nodes_ordered = zip(
        *sorted(
            zip(submeshes, nodes), key=functools.cmp_to_key(lambda x, y: _cmp_zorder(x[1], y[1]))  # type: ignore # pylint: disable=line-too-long
        )
    )
    # convert back from trimesh to CV Mesh class
    submeshes_ordered_cv = [Mesh(m.vertices, m.faces) for m in submeshes_ordered]

    return (submeshes_ordered_cv, nodes_ordered)
