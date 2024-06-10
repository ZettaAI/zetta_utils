# pylint: disable=line-too-long, unused-import, too-many-return-statements, unused-argument, redefined-outer-name
from __future__ import annotations

import filecmp
import os
import shutil

import attrs
import pytest

import zetta_utils
from zetta_utils import builder, mazepa, parsing
from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex, VolumetricLayer
from zetta_utils.layer.volumetric.cloudvol import build_cv_layer
from zetta_utils.layer.volumetric.precomputed.precomputed import _info_cache
from zetta_utils.layer.volumetric.tensorstore import TSBackend
from zetta_utils.mazepa_layer_processing.common import build_subchunkable_apply_flow

zetta_utils.load_all_modules()


@builder.register("OnlyCopyTempOp")
@mazepa.taskable_operation_cls
class OnlyCopyTempOp:
    def get_input_resolution(self, dst_resolution):  # pylint: disable=no-self-use
        return dst_resolution

    def with_added_crop_pad(self, crop_pad: Vec3D[int]):
        return self

    def __call__(
        self,
        idx: VolumetricIndex,
        dst: VolumetricLayer,
        src: VolumetricLayer,
    ) -> None:
        if dst.backend.name.find("_temp_") > 0:
            dst[idx] = src[idx]


@pytest.fixture
def clear_temp_dir_and_info_cache():
    temp_dir = "./assets/temp/"
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
    _info_cache.clear()


# from https://stackoverflow.com/questions/4187564/recursively-compare-two-directories-to-ensure-they-have-the-same-files-and-subdi
def are_dir_trees_equal(dir1, dir2):
    """
    Compare two directories recursively. Files in each directory are
    assumed to be equal if their names and contents are equal.

    @param dir1: First directory path
    @param dir2: Second directory path

    @return: True if the directory trees are the same and
        there were no errors while accessing the directories or files,
        or if the directory does not exist for both paths.
        False otherwise.
    """
    if not os.path.exists(dir1) and not os.path.exists(dir2):
        return True
    dirs_cmp = filecmp.dircmp(dir1, dir2)
    if len(dirs_cmp.left_only) > 0:
        print(f"File list mismatch: {dir1} has {dirs_cmp.left_only} files not found in {dir2}.")
        return False
    if len(dirs_cmp.right_only) > 0:
        print(f"File list mismatch: {dir2} has {dirs_cmp.right_only} files not found in {dir1}.")
        return False
    if len(dirs_cmp.funny_files) > 0:
        print(f"Cannot compare files: {dirs_cmp.funny_files}")
        return False
    (_, mismatch, errors) = filecmp.cmpfiles(dir1, dir2, dirs_cmp.common_files, shallow=False)
    if len(mismatch) > 0:
        print(f"Mismatched files: {mismatch}")
        return False
    if len(errors) > 0:
        print(f"Errors in comparing files: {errors}")
        return False
    for common_dir in dirs_cmp.common_dirs:
        new_dir1 = os.path.join(dir1, common_dir)
        new_dir2 = os.path.join(dir2, common_dir)
        if not are_dir_trees_equal(new_dir1, new_dir2):
            return False
    return True


@pytest.mark.skipif(
    "not config.getoption('--run-integration')",
    reason="Only run when `--run-integration` is given",
)
@pytest.mark.parametrize(
    "cue_name",
    [
        "test_uint8_no_dst",
        "test_uint8_copy_bbox",
        "test_uint8_copy_auto_bbox",
        "test_uint8_copy_dst_tighten_bounds",
        "test_uint8_copy_no_op_kwargs",
        "test_uint8_copy_coords",
        "test_uint8_copy_fn_semaphores",
        "test_uint8_copy_expand_bbox_resolution",
        "test_uint8_copy_expand_bbox_processing",
        "test_uint8_copy_expand_bbox_backend",
        "test_uint8_copy_expand_bbox_resolution_backend_processing_do_nothing",
        "test_uint8_copy_processing_gap",
        "test_uint8_copy_shrink_processing_chunk",
        "test_uint8_copy_op",
        "test_uint8_copy_auto_divisibility",
        "test_uint8_copy_skip_intermediaries",
        "test_uint8_copy_dont_skip_intermediaries",
        "test_uint8_copy_multilevel_no_checkerboard",
        "test_uint8_copy_multilevel_checkerboard",
        "test_uint8_copy_multilevel_checkerboard_cache_up_to_l0",
        "test_uint8_copy_blend",
        "test_uint8_copy_crop",
        "test_uint8_copy_defer",
        "test_uint8_copy_top_level_checkerboard",
        "test_uint8_copy_writeproc",
        "test_uint8_copy_writeproc_multilevel_no_checkerboard",
        "test_uint8_copy_writeproc_multilevel_checkerboard",
        "test_float32_copy",
        "test_float32_copy_multilevel_no_checkerboard",
        "test_float32_copy_multilevel_checkerboard",
        "test_float32_copy_blend",
        "test_float32_copy_crop",
        "test_float32_copy_writeproc_multilevel_no_checkerboard",
        "test_float32_copy_writeproc_multilevel_checkerboard",
        "test_float32_copy_writeproc_multilevel_checkerboard_parallel",
    ],
)
def test_subchunkable(cue_name, clear_temp_dir_and_info_cache):
    cue_path = f"./subchunkable/specs/{cue_name}.cue"
    ref_path = f"./assets/outputs_ref/{cue_name}"
    out_path = f"./assets/outputs/{cue_name}"
    spec = zetta_utils.parsing.cue.load(cue_path)
    zetta_utils.builder.build(spec)
    assert are_dir_trees_equal(ref_path, out_path)
    del spec


@pytest.mark.skipif(
    "not config.getoption('--run-integration')",
    reason="Only run when `--run-integration` is given",
)
@pytest.mark.parametrize(
    "cue_name",
    [
        "test_uint8_exc_no_bbox_or_coords_or_auto_bbox",
        "test_uint8_exc_both_bbox_and_coords",
        "test_uint8_exc_both_bbox_and_auto_bbox",
        "test_uint8_exc_both_coords_and_auto_bbox",
        "test_uint8_exc_no_fn_or_op",
        "test_uint8_exc_both_fn_and_op",
        "test_uint8_exc_both_fn_semaphores_and_op",
        "test_uint8_exc_seq_of_seq_not_equal",
        "test_uint8_exc_generate_ng_link_but_not_print_summary",
        "test_uint8_exc_level_intermediaries_dirs_not_equal",
        "test_uint8_exc_skip_intermediaries_but_level_intermediaries_dirs",
        "test_uint8_exc_skip_intermediaries_but_blend_pad",
        "test_uint8_exc_skip_intermediaries_but_crop_pad",
        "test_uint8_exc_skip_intermediaries_but_defer",
        "test_uint8_exc_not_skip_intermediaries_but_no_level_intermediaries_dirs",
        "test_uint8_exc_shrink_processing_chunk_and_expand_bbox_processing",
        "test_uint8_exc_bbox_non_integral_without_expand_bbox_resolution",
        "test_uint8_exc_bbox_non_integral_without_expand_bbox_resolution_but_expand_bbox_processing",
        "test_uint8_exc_bbox_non_integral_without_expand_bbox_resolution_but_shrink_processing_chunk",
        "test_uint8_exc_auto_divisibility_and_shrink_processing_chunk",
        "test_uint8_exc_auto_divisibility_but_no_expand_bbox_processing",
        "test_uint8_exc_auto_divisibility_and_expand_bbox_backend",
        "test_uint8_exc_processing_gap_but_auto_divisibility",
        "test_uint8_exc_processing_gap_but_blend_pad_toplevel",
        "test_uint8_exc_processing_gap_but_expand_bbox_backend",
        "test_uint8_exc_processing_gap_but_expand_bbox_resolution",
        "test_uint8_exc_processing_gap_but_shrink_processing_chunk",
        "test_uint8_exc_processing_gap_but_uneven",
        "test_uint8_exc_max_reduction_chunk_size_too_small_for_backend_chunk",
        "test_uint8_exc_max_reduction_chunk_size_too_small_for_processing_chunk",
        "test_uint8_exc_defer_on_not_toplevel",
        "test_uint8_exc_no_dst_but_defer",
        "test_uint8_exc_no_dst_but_max_reduction_chunk_size",
        "test_uint8_exc_no_dst_but_not_skip_intermediaries",
        "test_uint8_exc_no_dst_but_expand_bbox_backend",
        "test_uint8_exc_blend_too_large",
        "test_uint8_exc_defer_on_not_toplevel",
        "test_uint8_exc_defer_but_skip_intermediaries",
        "test_uint8_exc_nondivisible_but_recommendable",
        "test_uint8_exc_nondivisible_and_not_recommendable",
    ],
)
def test_subchunkable_val_exc(cue_name, clear_temp_dir_and_info_cache):
    cue_path = f"./subchunkable/specs/exc/{cue_name}.cue"
    spec = zetta_utils.parsing.cue.load(cue_path)
    with pytest.raises(ValueError):
        zetta_utils.builder.build(spec)
    del spec


COLLECTED_CHUNK_IDS = []


@mazepa.taskable_operation_cls
@attrs.frozen
class CollectChunkIDsOp:
    def get_input_resolution(self, dst_resolution: Vec3D) -> Vec3D:  # pylint: disable=no-self-use
        return dst_resolution

    def with_added_crop_pad(self, crop_pad: Vec3D[int]) -> CollectChunkIDsOp:
        return self

    def __call__(self, idx: VolumetricIndex, dst: VolumetricLayer, *args, **kwargs) -> None:
        COLLECTED_CHUNK_IDS.append(idx.chunk_id)


@pytest.mark.skipif(
    "not config.getoption('--run-integration')",
    reason="Only run when `--run-integration` is given",
)
def test_subchunkable_chunk_ids(clear_temp_dir_and_info_cache):
    global COLLECTED_CHUNK_IDS  # pylint: disable=global-statement
    COLLECTED_CHUNK_IDS = []
    dst = None
    dst_resolution = [4, 4, 40]
    processing_chunk_sizes = [[512, 512, 1], [256, 256, 1], [64, 64, 1]]
    bbox = BBox3D.from_coords(
        start_coord=Vec3D(0, 00, 100), end_coord=Vec3D(512, 512, 101), resolution=dst_resolution
    )
    flow = build_subchunkable_apply_flow(
        dst,
        dst_resolution,
        processing_chunk_sizes,
        bbox=bbox,
        skip_intermediaries=True,
        op=CollectChunkIDsOp(),
    )

    mazepa.execute(flow)

    COLLECTED_CHUNK_IDS.sort()
    assert COLLECTED_CHUNK_IDS == list(range(0, 64))


@pytest.mark.skipif(
    "not config.getoption('--run-integration')",
    reason="Only run when `--run-integration` is given",
)
def test_subchunkable_padded_chunk_ids(clear_temp_dir_and_info_cache):
    global COLLECTED_CHUNK_IDS  # pylint: disable=global-statement
    COLLECTED_CHUNK_IDS = []
    dst_resolution = [16, 16, 42]
    bbox = BBox3D.from_coords(
        start_coord=Vec3D(14250, 9800, 3060),
        end_coord=Vec3D(14762, 10312, 3092),
        resolution=dst_resolution,
    )
    dst = build_cv_layer(
        path="file://assets/temp/",
        on_info_exists="overwrite",
        info_field_overrides={
            "type": "image",
            "data_type": "int32",
            "num_channels": 1,
            "scales": [
                {
                    "encoding": "raw",
                    "resolution": dst_resolution,
                    "size": list(bbox.shape),
                    "chunk_sizes": [[256, 256, 32]],
                    "voxel_offset": list(bbox.start),
                    "key": "16_16_42",
                }
            ],
        },
    )
    processing_chunk_sizes = [[512, 512, 32], [320, 320, 32], [224, 224, 32]]
    flow = build_subchunkable_apply_flow(
        dst,
        dst_resolution,
        processing_chunk_sizes,
        bbox=bbox,
        processing_blend_pads=[[64, 64, 0], [64, 64, 0], [64, 64, 0]],
        skip_intermediaries=False,
        level_intermediaries_dirs=[
            "assets/temp/",
            "assets/temp/",
            "assets/temp/",
        ],
        max_reduction_chunk_size=[512, 512, 40],
        op=CollectChunkIDsOp(),
    )

    mazepa.execute(flow)

    COLLECTED_CHUNK_IDS.sort()
    assert COLLECTED_CHUNK_IDS == list(range(0, 16))
