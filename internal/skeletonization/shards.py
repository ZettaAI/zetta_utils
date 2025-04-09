# pylint: disable = c-extension-no-member
from __future__ import annotations

import math
import os
from collections import defaultdict
from time import strftime

import kimimaro
import numpy as np
import shardcomputer
from cloudfiles import CloudFile, CloudFiles, paths
from cloudvolume import CloudVolume, Skeleton
from cloudvolume.datasource.precomputed.sharding import (
    ShardingSpecification,
    synthesize_shard_files,
)
from cloudvolume.lib import sip
from mapbuffer import MapBuffer
from tqdm import tqdm

from zetta_utils import builder, log, mazepa
from zetta_utils.layer.db_layer import DBLayer

## Most of this file is a direct port of the
## sharded mesh generation in igneous to zetta_utils:
## https://github.com/seung-lab/igneous

logger = log.get_logger("zetta_utils")


@builder.register("build_make_skeleton_shards_flow")
def build_make_skeleton_shards_flow(
    segmentation_path: str,
    seg_db: DBLayer,
    frag_db: DBLayer,
    skeleton_dir: str,
    shard_index_bytes: int = 2 ** 13,
    minishard_index_bytes: int = 2 ** 15,
    min_shards: int = 1,
    num_shard_no_tasks: int = 1,
    frag_path: str | None = None,
    minishard_index_encoding: str = "gzip",
    max_seg_ids_per_shard: int | None = None,
    data_encoding: str = "gzip",
    max_cable_length: float | None = None,
    dust_threshold: int = 4000,
    tick_threshold: float = 6000,
) -> mazepa.Flow:
    """
    Wrapper that makes a Flow to make shards; see MakeSkeletonShardsFlowSchema.
    """
    return MakeSkeletonShardsFlowSchema()(
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
        data_encoding=data_encoding,
        max_cable_length=max_cable_length,
        dust_threshold=dust_threshold,
        tick_threshold=tick_threshold,
    )


@builder.register("MakeSkeletonShardsFlowSchema")
@mazepa.flow_schema_cls
class MakeSkeletonShardsFlowSchema:
    def flow(  # pylint: disable=too-many-locals, no-self-use
        self,
        segmentation_path: str,
        seg_db: DBLayer,
        frag_db: DBLayer,
        shard_index_bytes: int,
        minishard_index_bytes: int,
        minishard_index_encoding: str,
        min_shards: int,
        num_shard_no_tasks: int,
        skeleton_dir: str,
        frag_path: str | None,
        max_seg_ids_per_shard: int | None,
        data_encoding: str,
        max_cable_length: float | None,
        dust_threshold: int,
        tick_threshold: int,
    ) -> mazepa.FlowFnReturnType:
        """
        Combines skeletons from multiple fragments and stores it into shards.
        Wrapped as a FlowSchema and a build function to avoid the mazepa id generation
        issue; please see the build function for default values.

        :param segmentation_path: The segmentation path.
        :param seg_db: The DBLayer that contains each segment id - only used to keep
            track of the number of segments (necessary for sharding parameter computation).
        :param frag_db: The DBLayer that contains each segment id and fragment name, for
            all segids in each fragment. The flow will generate tasks to iterate through
            this database and assign each segid to a shard, and then will reference the
            assigned shard number to figure out which segids belong to the shard.
        :param skeleton_dir: The skeleton directory for the run; the fragments will be in
            `{frag_path}/{skeleton_dir}`, and the shards will be in
            `{segmentation_path}/{skeleton_dir}`.
            Note that if the segmentation infofile already has the mesh directory information,
            this parameter _must_ either not be provided or agree with the existing information.
        :param frag_path: Where the fragments are stored (with the `mesh_dir` prefix on
            the filenames). If not provided, defaults to the segmentation folder.
        :param shard_index_bytes: Sharding parameter; see `_compute_shard_params_for_hashed`.
        :param minishard_index_bytes: Sharding parameter; see `_compute_shard_params_for_hashed`.
        :param min_shards: Sharding parameter; see `_compute_shard_params_for_hashed`.
        :param num_shard_no_tasks: Number of tasks to generate for assigning shard numbers
            to mesh fragments in the `frag_db`.
        :param minishard_index_encoding: Minishard index encoding; see
            "https://github.com/seung-lab/igneous/blob/
            7258212e57c5cfc1e5f0de8162f830c7d49e1be9/igneous/task_creation/image.py#L343"
            for details.
        :param max_seg_ids_per_shard: Max number of segment ids that can be assigned
            to a single shard.
        :param data_encoding: Encoding for shard files.
        :param max_cable_length: Sharding parameter, see kimimaro documentation.
        :param dust_threshold: Skip segments smaller than this number in voxels.
        :param tick_threshold: Keep ticks larger than this number in nanometres.
        """

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
            data_encoding=data_encoding,
        )

        cv = CloudVolume(segmentation_path)
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

        cv.skeleton.meta.info["sharding"] = spec.to_dict()
        cv.skeleton.meta.commit_info()
        # rebuild b/c sharding changes the skeleton source class
        cv = CloudVolume(segmentation_path)
        cv.mip = cv.skeleton.meta.mip

        # assign shard numbers distributedly
        batch_size = math.ceil(len(frag_db) / num_shard_no_tasks)
        shard_no_tasks = [
            _compute_shard_no_for_fragments.make_task(
                frag_db, batch_num, batch_size, preshift_bits, shard_bits, minishard_bits
            )
            for batch_num in range(num_shard_no_tasks)
        ]
        yield shard_no_tasks
        yield mazepa.Dependency()
        for task in shard_no_tasks:
            assert task.outcome is not None
            assert task.outcome.return_value is not None
        shard_numbers = set().union(*(task.outcome.return_value for task in shard_no_tasks))  # type: ignore # pylint: disable=line-too-long
        cv.provenance.processing.append(
            {
                "method": {
                    "task": "ShardedSkeletonMergeTask",
                    "segmentation_path": segmentation_path,
                    "mip": cv.skeleton.meta.mip,
                    "skeleton_dir": skeleton_dir_to_use,
                    "frag_path": frag_path,
                    "dust_threshold": dust_threshold,
                    "tick_threshold": tick_threshold,
                    "max_cable_length": max_cable_length,
                    "preshift_bits": preshift_bits,
                    "minishard_bits": minishard_bits,
                    "shard_bits": shard_bits,
                },
                "by": "zetta_utils",
                "date": strftime("%Y-%m-%d %H:%M %Z"),
            }
        )
        cv.commit_provenance()

        tasks = [
            _multires_sharded_skeleton_merge.make_task(
                segmentation_path,
                frag_db,
                shard_no,
                skeleton_dir=skeleton_dir,
                frag_path=frag_path,
                max_cable_length=max_cable_length,
                dust_threshold=dust_threshold,
                tick_threshold=tick_threshold,
            )
            for shard_no in shard_numbers
        ]

        yield tasks
        yield mazepa.Dependency()
        for task in tasks:
            assert task.outcome is not None


# TODO: Refactor duplicate with Mesh Sharding, alongside where this is called
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
def _multires_sharded_skeleton_merge(
    segmentation_path: str,
    frag_db: DBLayer,
    shard_no: str,
    skeleton_dir: str,
    max_cable_length: float | None,
    dust_threshold: int,
    tick_threshold: int,
    frag_path: str | None = None,
):
    """
    Creates and uploads the multires shard of the given shard number
    by merging skeleton fragments for seg ids in the shard and creating the LODs required.

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
    cv.mip = cv.skeleton.meta.mip

    query_results = frag_db.query({"shard_no": [shard_no]}, return_columns=("seg_id", "frag_fn"))
    # TODO: return type is a little too broad?
    seg_ids = list(set(int(query_result["seg_id"]) for query_result in query_results.values()))  # type: ignore # pylint: disable=line-too-long
    filenames = list(set(str(query_result["frag_fn"]) for query_result in query_results.values()))

    skeletons = _get_unfused_skeletons(cv, seg_ids, filenames, skeleton_dir, frag_path)
    del seg_ids
    del filenames
    skeletons = _process_skeletons(
        skeletons, max_cable_length, dust_threshold, tick_threshold, in_place=True
    )

    if len(skeletons) == 0:
        return

    shard_files = synthesize_shard_files(cv.skeleton.reader.spec, skeletons)

    if len(shard_files) != 1:
        raise ValueError(
            "Only one shard file should be generated per task. "
            f"Expected: {shard_no} Got: {', '.join(shard_files.keys())}"
        )

    cf = CloudFiles(cv.skeleton.meta.layerpath)
    cf.puts(
        ((fname, data) for fname, data in shard_files.items()),
        compress=False,
        content_type="application/octet-stream",
        cache_control="no-cache",
    )


def _get_unfused_skeletons(
    cv: CloudVolume,
    seg_ids: list[int],
    filenames: list[str],
    skeleton_dir: str,
    frag_path: str | None,
):
    filenames = [cv.meta.join(skeleton_dir, loc) for loc in filenames]

    block_size = 50

    if len(filenames) < block_size:
        blocks = [filenames]
        n_blocks = 1
    else:
        n_blocks = max(len(filenames) // block_size, 1)
        blocks = sip(filenames, block_size)

    frag_prefix = frag_path or cv.cloudpath
    local_input = False
    if paths.extract(frag_prefix).protocol == "file":
        local_input = True
        frag_prefix = frag_prefix.replace("file://", "", 1)

    all_skels = defaultdict(list)
    for filenames_block in tqdm(blocks, desc="Filename Block", total=n_blocks):
        if local_input:
            all_files = {}
            for filename in filenames_block:
                all_files[filename] = open(  # pylint: disable = consider-using-with
                    os.path.join(frag_prefix, filename),
                    "rb",
                )
        else:
            all_files = {
                filename: CloudFile(cv.meta.join(frag_prefix, filename), cache_meta=True)
                for filename in filenames_block
            }

        for filename, content in tqdm(all_files.items(), desc="Scanning Fragments"):
            fragment = MapBuffer(content, frombytesfn=Skeleton.from_precomputed)

            for seg_id in seg_ids:
                try:
                    skel = fragment[seg_id]
                    skel.id = seg_id
                    all_skels[seg_id].append(skel)
                except KeyError:
                    continue

                if hasattr(content, "close"):
                    content.close()

    return all_skels


def _process_skeletons(
    unfused_skeletons, max_cable_length, dust_threshold, tick_threshold, in_place=False
):
    skeletons = {}
    if in_place:
        skeletons = unfused_skeletons

    for label in tqdm(unfused_skeletons.keys(), desc="Postprocessing"):
        skels = unfused_skeletons[label]
        skel = Skeleton.simple_merge(skels)
        skel.id = label
        skel.extra_attributes = [
            attr for attr in skel.extra_attributes if attr["data_type"] == "float32"
        ]
        skel = skel.consolidate()

        if max_cable_length is not None and skel.cable_length() > max_cable_length:
            skeletons[label] = skel.to_precomputed()
        else:
            skeletons[label] = kimimaro.postprocess(
                skel,
                dust_threshold=dust_threshold,  # voxels
                tick_threshold=tick_threshold,  # nm
            ).to_precomputed()

    return skeletons
