# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,too-few-public-methods
import os
import pathlib

import numpy as np
import pytest

from zetta_utils.bcube import BoundingCube
from zetta_utils.layer.volumetric import VolumetricIndex, cloudvol
from zetta_utils.layer.volumetric.cloudvol import CVBackend

THIS_DIR = pathlib.Path(__file__).parent.resolve()
INFOS_DIR = THIS_DIR / "../../../assets/infos/"
LAYER_X0_PATH = os.path.join(INFOS_DIR, "layer_x0")
LAYER_X1_PATH = os.path.join(INFOS_DIR, "layer_x1")


@pytest.fixture
def clear_caches():
    cloudvol.backend._cv_cache.clear()


def test_cv_backend_specific_mip_exc():
    with pytest.raises(ValueError):
        CVBackend(path="", cv_kwargs={"mip": 4})


def test_cv_backend_bad_path_exc():
    with pytest.raises(ValueError):
        CVBackend(path="abc")


def test_cv_backend_info_expect_same_exc(mocker):
    cv_m = mocker.MagicMock()
    mocker.patch(
        "cloudvolume.CloudVolume.__new__",
        return_value=cv_m,
    )
    cv_m.commit_info = mocker.MagicMock()

    info_spec = cloudvol.backend.PrecomputedInfoSpec(
        reference_path=LAYER_X0_PATH,
    )
    with pytest.raises(RuntimeError):
        CVBackend(path=LAYER_X1_PATH, info_spec=info_spec, on_info_exists="expect_same")
    cv_m.commit_info.assert_not_called()


@pytest.mark.parametrize(
    "path, reference, mode",
    [
        [LAYER_X0_PATH, LAYER_X0_PATH, "overwrite"],
        [LAYER_X0_PATH, LAYER_X0_PATH, "expect_same"],
        [LAYER_X0_PATH, None, "overwrite"],
        [LAYER_X0_PATH, None, "expect_same"],
    ],
)
def test_cv_backend_info_no_action(path, reference, mode, mocker):
    cv_m = mocker.MagicMock()
    mocker.patch(
        "cloudvolume.CloudVolume.__new__",
        return_value=cv_m,
    )
    cv_m.commit_info = mocker.MagicMock()
    info_spec = cloudvol.backend.PrecomputedInfoSpec(
        reference_path=reference,
    )
    CVBackend(path=path, info_spec=info_spec, on_info_exists=mode)

    cv_m.commit_info.assert_not_called()


@pytest.mark.parametrize(
    "path, reference, mode",
    [
        [LAYER_X1_PATH, LAYER_X0_PATH, "overwrite"],
        [".", LAYER_X0_PATH, "overwrite"],
        [".", LAYER_X0_PATH, "expect_same"],
    ],
)
def test_cv_backend_info_overwrite(path, reference, mode, mocker):
    cv_m = mocker.MagicMock()
    mocker.patch(
        "cloudvolume.CloudVolume.__new__",
        return_value=cv_m,
    )
    cv_m.commit_info = mocker.MagicMock()
    info_spec = cloudvol.backend.PrecomputedInfoSpec(
        reference_path=reference,
        chunk_size=[1024, 1024, 1],
    )
    CVBackend(path=path, info_spec=info_spec, on_info_exists=mode)

    cv_m.commit_info.assert_called_once()


def test_cv_backend_read(clear_caches, mocker):
    data_read = np.ones([3, 4, 5, 2])
    expected = np.ones([2, 3, 4, 5])
    cv_m = mocker.MagicMock()
    cv_m.__getitem__ = mocker.MagicMock(return_value=data_read)
    mocker.patch("cloudvolume.CloudVolume.__new__", return_value=cv_m)
    cvb = CVBackend(path="path")
    index = VolumetricIndex(
        bcube=BoundingCube.from_slices((slice(0, 1), slice(1, 2), slice(2, 3))),
        resolution=(1, 1, 1),
    )
    result = cvb.read(index)
    np.testing.assert_array_equal(result, expected)
    cv_m.__getitem__.assert_called_with(index.bcube.to_slices(index.resolution))


def test_cv_backend_write(clear_caches, mocker):
    cv_m = mocker.MagicMock()
    cv_m.__setitem__ = mocker.MagicMock()
    mocker.patch("cloudvolume.CloudVolume.__new__", return_value=cv_m)
    cvb = CVBackend(path="path")
    value = np.ones([2, 3, 4, 5])
    expected_written = np.ones([3, 4, 5, 2])  # channel as ch 0

    index = VolumetricIndex(
        bcube=BoundingCube.from_slices((slice(0, 1), slice(1, 2), slice(2, 3))),
        resolution=(1, 1, 1),
    )
    cvb.write(index, value)
    assert cv_m.__setitem__.call_args[0][0] == index.bcube.to_slices(index.resolution)
    np.testing.assert_array_equal(
        cv_m.__setitem__.call_args[0][1],
        expected_written,
    )


def test_cv_backend_write_scalar(clear_caches, mocker):
    cv_m = mocker.MagicMock()
    cv_m.__setitem__ = mocker.MagicMock()
    mocker.patch("cloudvolume.CloudVolume.__new__", return_value=cv_m)
    cvb = CVBackend(path="path")
    value = 1
    expected_written = 1

    index = VolumetricIndex(
        bcube=BoundingCube.from_slices((slice(0, 1), slice(1, 2), slice(2, 3))),
        resolution=(1, 1, 1),
    )
    cvb.write(index, value)
    assert cv_m.__setitem__.call_args[0][0] == index.bcube.to_slices(index.resolution)
    np.testing.assert_array_equal(
        cv_m.__setitem__.call_args[0][1],
        expected_written,
    )


@pytest.mark.parametrize(
    "data_in,expected_exc",
    [
        # Too many dims
        [np.ones((1, 2, 3, 4, 5, 6)), ValueError],
    ],
)
def test_cv_backend_write_exc(data_in, expected_exc, clear_caches, mocker):
    cv_m = mocker.MagicMock()
    cv_m.__setitem__ = mocker.MagicMock()
    mocker.patch("cloudvolume.CloudVolume.__new__", return_value=cv_m)
    cvb = CVBackend(path="path")
    index = VolumetricIndex(
        bcube=BoundingCube.from_slices((slice(1, 1), slice(1, 2), slice(2, 3))),
        resolution=(1, 1, 1),
    )
    with pytest.raises(expected_exc):
        cvb.write(index, data_in)
