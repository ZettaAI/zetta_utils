# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,too-few-public-methods
import os
import pathlib

import pytest

from zetta_utils.geometry import IntVec3D, Vec3D
from zetta_utils.layer.volumetric.precomputed import (
    PrecomputedInfoSpec,
    get_info,
    precomputed,
)

THIS_DIR = pathlib.Path(__file__).parent.resolve()
INFOS_DIR = os.path.abspath(THIS_DIR / "../../../assets/infos/")
LAYER_X0_PATH = "file://" + os.path.join(INFOS_DIR, "layer_x0")
LAYER_X1_PATH = "file://" + os.path.join(INFOS_DIR, "layer_x1")
LAYER_X2_PATH = "file://" + os.path.join(INFOS_DIR, "layer_x2")
LAYER_X3_PATH = "file://" + os.path.join(INFOS_DIR, "layer_x3")
LAYER_X4_PATH = "file://" + os.path.join(INFOS_DIR, "layer_x4")
LAYER_X5_PATH = "file://" + os.path.join(INFOS_DIR, "scratch", "layer_x5")
LAYER_X6_PATH = "file://" + os.path.join(INFOS_DIR, "scratch", "layer_x6")
LAYER_X7_PATH = "file://" + os.path.join(INFOS_DIR, "scratch", "layer_x7")

_write_info_notmock = precomputed._write_info


@pytest.fixture
def clear_caches():
    precomputed._info_cache.clear()


def test_infospec_expect_same_exc(mocker):
    precomputed._write_info = mocker.MagicMock()
    info_spec = PrecomputedInfoSpec(
        reference_path=LAYER_X0_PATH,
    )
    with pytest.raises(RuntimeError):
        info_spec.update_info(path=LAYER_X1_PATH, on_info_exists="expect_same")
    precomputed._write_info.assert_not_called()


@pytest.mark.parametrize(
    "path, reference, mode",
    [
        [LAYER_X0_PATH, LAYER_X0_PATH, "overwrite"],
        [LAYER_X0_PATH, LAYER_X0_PATH, "expect_same"],
        [LAYER_X0_PATH, None, "overwrite"],
        [LAYER_X0_PATH, None, "expect_same"],
    ],
)
def test_infospec_no_action(path, reference, mode, mocker):
    precomputed._write_info = mocker.MagicMock()
    info_spec = PrecomputedInfoSpec(
        reference_path=reference,
    )
    info_spec.update_info(path=path, on_info_exists=mode)

    precomputed._write_info.assert_not_called()


@pytest.mark.parametrize(
    "path, reference, mode",
    [
        [LAYER_X1_PATH, LAYER_X0_PATH, "overwrite"],
        [".", LAYER_X0_PATH, "overwrite"],
    ],
)
def test_infospec_overwrite(clear_caches, path, reference, mode, mocker):
    precomputed._write_info = mocker.MagicMock()
    info_spec = PrecomputedInfoSpec(
        reference_path=reference,
        default_chunk_size=IntVec3D(999, 999, 1),
    )
    info_spec.update_info(path=path, on_info_exists=mode)

    precomputed._write_info.assert_called_once()


def test_ts_set_voxel_set_voxel_offset_chunk_and_data(clear_caches, mocker):
    precomputed._write_info = _write_info_notmock
    info_spec = PrecomputedInfoSpec(
        reference_path=LAYER_X0_PATH,
        default_chunk_size=IntVec3D(1024, 1024, 1),
    )
    info_spec.set_chunk_size((IntVec3D(3, 2, 1), Vec3D(2, 2, 1)))
    info_spec.set_voxel_offset((IntVec3D(1, 2, 3), Vec3D(2, 2, 1)))
    info_spec.set_dataset_size((IntVec3D(10, 20, 30), Vec3D(2, 2, 1)))
    info_spec.update_info(LAYER_X5_PATH, on_info_exists="overwrite")
    scales = get_info(LAYER_X5_PATH)["scales"]
    for scale in scales:
        if scale["resolution"] == [2, 2, 1]:
            assert scale["voxel_offset"] == [1, 2, 3]
            assert scale["chunk_sizes"][0] == [3, 2, 1]
            assert scale["size"] == [10, 20, 30]
