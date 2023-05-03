# pylint: disable=line-too-long, unused-import, too-many-return-statements
import filecmp
import os

import pytest

import zetta_utils
from zetta_utils import builder, parsing
from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex
from zetta_utils.layer.volumetric.tensorstore import TSBackend


# from https://stackoverflow.com/questions/4187564/recursively-compare-two-directories-to-ensure-they-have-the-same-files-and-subdi
def are_dir_trees_equal(dir1, dir2):
    """
    Compare two directories recursively. Files in each directory are
    assumed to be equal if their names and contents are equal.

    @param dir1: First directory path
    @param dir2: Second directory path

    @return: True if the directory trees are the same and
        there were no errors while accessing the directories or files,
        False otherwise.
    """

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
        "test_uint8_copy",
        "test_uint8_copy_multilevel",
        "test_uint8_copy_blend",
        "test_uint8_copy_crop",
        "test_uint8_copy_writeproc",
        "test_uint8_copy_writeproc_multilevel",
        "test_float32_copy",
        "test_float32_copy_multilevel",
        "test_float32_copy_blend",
        "test_float32_copy_crop",
        "test_float32_copy_writeproc",
        "test_float32_copy_writeproc_multilevel",
    ],
)
def test_subchunkable(cue_name):
    cue_path = f"./tests/integration/subchunkable/specs/{cue_name}.cue"
    ref_path = f"./tests/integration/assets/outputs_ref/{cue_name}"
    out_path = f"./tests/integration/assets/outputs/{cue_name}"
    spec = zetta_utils.parsing.cue.load(cue_path)
    zetta_utils.builder.build(spec)
    assert are_dir_trees_equal(ref_path, out_path)
