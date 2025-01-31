# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,too-few-public-methods
import os
import pathlib

import pytest

from zetta_utils.geometry import IntVec3D
from zetta_utils.geometry.bbox import BBox3D
from zetta_utils.layer import precomputed
from zetta_utils.layer.precomputed import InfoSpecParams, PrecomputedInfoSpec, get_info

THIS_DIR = pathlib.Path(__file__).parent.resolve()
INFOS_DIR = os.path.abspath(THIS_DIR / "../assets/infos/")
LAYER_X0_PATH = "file://" + os.path.join(INFOS_DIR, "layer_x0")
LAYER_X1_PATH = "file://" + os.path.join(INFOS_DIR, "layer_x1")
LAYER_SCRATCH0_PATH = "file://" + os.path.join(INFOS_DIR, "scratch", "layer_x0")
NONEXISTENT_LAYER_PATH = "file://" + os.path.join(INFOS_DIR, "scratch", "nonexistent_layer")

EXAMPLE_INFO = '{"data_type": "uint8", "num_channels": 1, "scales": [{"chunk_sizes": [[1024, 1024, 1]], "encoding": "raw", "key": "4_4_40", "resolution": [4, 4, 40], "size": [16384, 16384, 16384], "voxel_offset": [0, 0, 0]}, {"chunk_sizes": [[2048, 2048, 1]], "encoding": "raw", "key": "8_8_40", "resolution": [8, 8, 40], "size": [8192, 8192, 8192], "voxel_offset": [0, 0, 0]}], "type": "raw"}'

_write_info_notmock = precomputed._write_info


@pytest.fixture
def clear_caches_reset_mocks():
    precomputed._info_cache.clear()
    precomputed._write_info = _write_info_notmock


def test_nonexistent_info(clear_caches_reset_mocks):
    with pytest.raises(FileNotFoundError):
        precomputed.get_info(NONEXISTENT_LAYER_PATH)


def test_infospec_expect_same_exc(clear_caches_reset_mocks, mocker):
    _write_info = mocker.MagicMock()
    precomputed._write_info = _write_info
    info_spec = PrecomputedInfoSpec(
        info_path=LAYER_X0_PATH,
    )
    with pytest.raises(RuntimeError):
        info_spec.update_info(path=LAYER_X1_PATH, on_info_exists="expect_same")
    _write_info.assert_not_called()


@pytest.mark.parametrize(
    "path, reference, mode",
    [
        [LAYER_X0_PATH, LAYER_X0_PATH, "overwrite"],
        [LAYER_X0_PATH, LAYER_X0_PATH, "expect_same"],
    ],
)
def test_infospec_no_action(clear_caches_reset_mocks, path, reference, mode, mocker):
    _write_info = mocker.MagicMock()
    precomputed._write_info = _write_info
    info_spec = PrecomputedInfoSpec(
        info_path=reference,
    )
    info_spec.update_info(path=path, on_info_exists=mode)

    _write_info.assert_not_called()


@pytest.mark.parametrize(
    "path, reference, mode",
    [
        [LAYER_X1_PATH, LAYER_X0_PATH, "overwrite"],
        [".", LAYER_X0_PATH, "overwrite"],
    ],
)
def test_infospec_overwrite(clear_caches_reset_mocks, path, reference, mode, mocker):
    _write_info = mocker.MagicMock()
    precomputed._write_info = _write_info
    info_spec = PrecomputedInfoSpec(
        info_spec_params=InfoSpecParams.from_reference(
            reference_path=reference,
            scales=[[1, 1, 1]],
            chunk_size=IntVec3D(999, 999, 1),
            inherit_all_params=True,
        )
    )
    info_spec.update_info(path=path, on_info_exists=mode)

    _write_info.assert_called_once()


@pytest.mark.parametrize(
    "path, reference",
    [
        [LAYER_X1_PATH, LAYER_X0_PATH],
        [LAYER_X1_PATH + "yo", LAYER_X0_PATH],
    ],
)
def test_infospec_extend(clear_caches_reset_mocks, path, reference, mocker):
    _write_info = mocker.MagicMock()
    precomputed._write_info = _write_info
    info_spec = PrecomputedInfoSpec(
        info_spec_params=InfoSpecParams.from_reference(
            reference_path=reference,
            chunk_size=IntVec3D(999, 999, 1),
            scales=[[128, 128, 1]],
            inherit_all_params=True,
        )
    )
    info_spec.update_info(path=path, on_info_exists="extend")
    _write_info.assert_called_once()


def test_rounding_scales(clear_caches_reset_mocks):
    info_spec = PrecomputedInfoSpec(
        info_spec_params=InfoSpecParams.from_reference(
            reference_path=LAYER_X0_PATH,
            scales=[[1, 1, 1], [2, 2, 1]],
            bbox=BBox3D.from_coords(
                start_coord=(1, 1, 0), end_coord=(4, 4, 4), resolution=[1, 1, 1]
            ),
            inherit_all_params=True,
        )
    )
    info = info_spec.make_info()
    assert info is not None
    assert info["scales"][0]["voxel_offset"] == [1, 1, 0]
    assert info["scales"][0]["size"] == [3, 3, 4]
    assert info["scales"][1]["voxel_offset"] == [0, 0, 0]
    assert info["scales"][1]["size"] == [2, 2, 4]


def test_type(clear_caches_reset_mocks):
    info_spec = PrecomputedInfoSpec(
        info_spec_params=InfoSpecParams.from_reference(
            reference_path=LAYER_X0_PATH,
            type="segmentation",
            inherit_all_params=True,
            scales=[[1, 1, 1]],
            extra_scale_data={},
        )
    )
    info = info_spec.make_info()
    assert info is not None
    assert info["type"] == "segmentation"


def test_data_type(clear_caches_reset_mocks):
    info_spec = PrecomputedInfoSpec(
        info_spec_params=InfoSpecParams.from_reference(
            reference_path=LAYER_X0_PATH,
            data_type="uint16",
            inherit_all_params=True,
            scales=[[1, 1, 1]],
        )
    )
    info = info_spec.make_info()
    assert info is not None
    assert info["data_type"] == "uint16"


def test_num_channels(clear_caches_reset_mocks):
    info_spec = PrecomputedInfoSpec(
        info_spec_params=InfoSpecParams.from_reference(
            reference_path=LAYER_X0_PATH,
            num_channels=4,
            inherit_all_params=True,
            scales=[[1, 1, 1]],
        )
    )
    info = info_spec.make_info()
    assert info is not None
    assert info["num_channels"] == 4


def test_set_voxel_offset_chunk_and_data(clear_caches_reset_mocks):
    info_spec = PrecomputedInfoSpec(
        info_spec_params=InfoSpecParams.from_reference(
            reference_path=LAYER_X0_PATH,
            chunk_size=[3, 2, 1],
            scales=[[1, 1, 1]],
            bbox=BBox3D.from_coords([1, 2, 3], [11, 22, 33], [1, 1, 1]),
            inherit_all_params=True,
        )
    )
    info_spec.update_info(LAYER_SCRATCH0_PATH, on_info_exists="overwrite")
    scales = get_info(LAYER_SCRATCH0_PATH)["scales"]
    for scale in scales:
        assert scale["voxel_offset"] == [1, 2, 3]
        assert scale["chunk_sizes"][0] == [3, 2, 1]
        assert scale["size"] == [10, 20, 30]


def test_constructor_underspecified_exc(clear_caches_reset_mocks):
    with pytest.raises(Exception):
        PrecomputedInfoSpec()


def test_constructor_overrspecified_exc(clear_caches_reset_mocks):
    info_spec_params = InfoSpecParams.from_reference(
        reference_path=LAYER_X0_PATH, scales=[[1, 1, 1]], inherit_all_params=True
    )
    with pytest.raises(Exception):
        PrecomputedInfoSpec(
            info_path=LAYER_X0_PATH,
            info_spec_params=info_spec_params,
        )


def test_no_inherit_exc(clear_caches_reset_mocks):
    with pytest.raises(Exception):
        InfoSpecParams.from_reference(
            reference_path=LAYER_X0_PATH,
            scales=[[1, 1, 1]],
        )


def test_no_scales_exc(clear_caches_reset_mocks):
    with pytest.raises(Exception):
        InfoSpecParams.from_reference(
            reference_path=LAYER_X0_PATH, scales=[], inherit_all_params=True
        )


def test_not_pure_extension_exc(clear_caches_reset_mocks, mocker):
    _write_info = mocker.MagicMock()
    precomputed._write_info = _write_info
    info_spec = PrecomputedInfoSpec(
        info_spec_params=InfoSpecParams.from_optional_reference(
            reference_path=LAYER_X0_PATH,
            chunk_size=[3, 2, 1],
            scales=[[1, 1, 1]],
            bbox=BBox3D.from_coords([1, 2, 3], [11, 22, 33], [1, 1, 1]),
            inherit_all_params=True,
            type="segmentation",
            data_type="int32",
        )
    )

    with pytest.raises(Exception):
        info_spec.update_info(LAYER_X0_PATH, "extend")
    _write_info.assert_not_called()


def test_change_scale_on_extend_exc(clear_caches_reset_mocks, mocker):
    _write_info = mocker.MagicMock()
    precomputed._write_info = _write_info
    info_spec = PrecomputedInfoSpec(
        info_spec_params=InfoSpecParams.from_optional_reference(
            reference_path=LAYER_X0_PATH,
            chunk_size=[7, 7, 7],
            scales=[[1, 1, 1]],
            inherit_all_params=True,
        )
    )
    with pytest.raises(Exception):
        info_spec.update_info(LAYER_X0_PATH, "extend")
    _write_info.assert_not_called()


def test_no_reference(clear_caches_reset_mocks):
    InfoSpecParams.from_optional_reference(
        reference_path=None,
        chunk_size=[3, 2, 1],
        scales=[[1, 1, 1]],
        bbox=BBox3D.from_coords([1, 2, 3], [11, 22, 33], [1, 1, 1]),
        data_type="uint8",
        type="segmentation",
        num_channels=1,
        encoding="raw",
    )


def test_no_reference_underspecified_exc(clear_caches_reset_mocks):
    with pytest.raises(Exception):
        InfoSpecParams.from_optional_reference(reference_path=None, scales=[[1, 1, 1]])


def test_no_reference_inherit_exc(clear_caches_reset_mocks):
    with pytest.raises(Exception):
        InfoSpecParams.from_optional_reference(
            reference_path=None,
            chunk_size=[3, 2, 1],
            scales=[[1, 1, 1]],
            bbox=BBox3D.from_coords([1, 2, 3], [11, 22, 33], [1, 1, 1]),
            data_type="uint8",
            type="segmentation",
            num_channels=1,
            encoding="raw",
            inherit_all_params=True,
        )
