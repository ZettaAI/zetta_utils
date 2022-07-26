# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,too-few-public-methods
import pytest
import numpy as np

from zetta_utils.data.layers.volumetric import translate_volumetric_index
from zetta_utils.data.layers.volumetric import _standardize_vol_idx
from zetta_utils.types import (
    VolumetricIndex,
    BoundingCube,
    Vec3D,
    Array,
    CVLayer,
    VolumetricLayer,
)



def test_volumetric_getitem(mocker):
    cvl = VolumetricLayer()
    std_idx = ((1, 1, 1), BoundingCube(slices=(slice(1, 1), slice(1, 1), slice(1, 1))))
    expected = np.array([5566])
    std_idx_fn = mocker.patch(
        "zetta_utils.data.layers.volumetric._standardize_vol_idx", return_value=std_idx
    )
    read_fn = mocker.patch(
        "zetta_utils.data.layers.volumetric.VolumetricLayer._read",
        return_value=expected,
    )
    result = cvl[0:0, 0:0, 0:0]
    np.testing.assert_array_equal(result, expected)
    std_idx_fn.assert_called_with((slice(0, 0), slice(0, 0), slice(0, 0)), index_resolution=None)
    read_fn.assert_called_with(idx=std_idx)


class DummyCV:
    def __init__(self, expected):
        self.expected = expected

    def __getitem__(self, *arg):
        return self.expected


@pytest.mark.parametrize(
    "data_read, dim_order, expected",
    [
        [np.ones((1, 2, 3, 1)), "cxyz", np.ones((1, 1, 2, 3))],
        [np.ones((1, 2, 3, 1)), "bcxyz", np.ones((1, 1, 1, 2, 3))],
        [np.ones((3, 2, 1, 1)), "cxy", np.ones((1, 3, 2))],
        [np.ones((3, 2, 1, 1)), "bcxy", np.ones((1, 1, 3, 2))],
    ],
)
def test_cvb_read(data_read, dim_order, expected, mocker):
    cvl = CVLayer({}, dim_order=dim_order)

    get_cv_fn = mocker.patch(
        "zetta_utils.data.layers.volumetric.CVLayer._get_cv_at_resolution",
        return_value=DummyCV(data_read),
    )

    std_idx = ((1, 1, 1), BoundingCube(slices=(slice(1, 1), slice(1, 1), slice(1, 1))))
    result = cvl._read_volume(std_idx)

    np.testing.assert_array_equal(result, expected)
    get_cv_fn.assert_called_with((1, 1, 1))


@pytest.mark.parametrize(
    "data_read, dim_order, expected_exc",
    [
        [np.ones((3, 2, 3, 1)), "cxy", RuntimeError],
    ],
)
def test_cvl_read_volume_exc(data_read, dim_order, expected_exc, mocker):
    cvl = CVLayer({}, dim_order=dim_order)

    mocker.patch(
        "zetta_utils.data.layers.volumetric.CVLayer._get_cv_at_resolution",
        return_value=DummyCV(data_read),
    )

    std_idx = ((1, 1, 1), BoundingCube(slices=(slice(1, 1), slice(1, 1), slice(1, 1))))

    with pytest.raises(expected_exc):
        cvl._read_volume(std_idx)


@pytest.mark.parametrize(
    "data_res, index_res, read_res, expected",
    [
        [(1, 1, 1), (2, 2, 2), (3, 3, 3), (1, 1, 1)],
        [(1, 1, 1), None, None, (1, 1, 1)],
        [None, (2, 2, 2), (3, 3, 3), (3, 3, 3)],
        [None, (2, 2, 2), None, (2, 2, 2)],
    ],
)
def test_cvl_read_correct_res(data_res, index_res, read_res, expected, mocker):
    cvl = CVLayer({}, data_resolution=data_res, index_resolution=index_res)

    bcube = BoundingCube(slices=(slice(1, 1), slice(1, 1), slice(1, 1)))

    idx = (read_res, bcube)

    _read_volume_fn = mocker.patch(
        "zetta_utils.data.layers.volumetric.CVLayer._read_volume",
        side_effect=SystemExit,
    )

    try:
        cvl._read(idx)
    except SystemExit:
        pass

    _read_volume_fn.assert_called_with((expected, bcube))


@pytest.mark.parametrize(
    "data_res, index_res, read_res, expected_exc",
    [
        [None, None, None, RuntimeError],
    ],
)
def test_cvl_read_correct_res_exc(data_res, index_res, read_res, expected_exc, mocker):
    cvl = CVLayer({}, data_resolution=data_res, index_resolution=index_res)

    bcube = BoundingCube(slices=(slice(1, 1), slice(1, 1), slice(1, 1)))

    idx = (read_res, bcube)

    with pytest.raises(RuntimeError):
        cvl._read(idx)


@pytest.mark.parametrize(
    "data_res, read_res, data_read, dim_order, expected",
    [
        [None, (2, 2, 2), np.ones((1, 1, 1, 2, 3)), "bcxyz", np.ones((1, 1, 1, 2, 3))],
        [
            (4, 4, 4),
            (2, 2, 2),
            np.ones((1, 1, 1, 2, 3)),
            "bcxyz",
            np.ones((1, 1, 2, 4, 6)),
        ],
        [(4, 4, 4), (2, 2, 2), np.ones((1, 1, 2, 3)), "bcxy", np.ones((1, 1, 4, 6))],
    ],
)
def test_cvl_read(data_res, read_res, data_read, dim_order, expected, mocker):
    cvl = CVLayer(
        {"cloudpath": "dummy"},
        data_resolution=data_res,
        dim_order=dim_order,
        interpolation_mode="img",
    )

    bcube = BoundingCube(slices=(slice(1, 1), slice(1, 1), slice(1, 1)))

    idx = (read_res, bcube)

    _read_volume_fn = mocker.patch(
        "zetta_utils.data.layers.volumetric.CVLayer._read_volume",
        return_value=data_read,
    )

    result = cvl._read(idx)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "data_res, read_res, data_read, dim_order, interp_mode, expected_exc",
    [
        [(4, 4, 4), (2, 2, 2), np.ones((1, 1, 2, 3)), "cxyz", "img", RuntimeError],
        [(4, 4, 4), (2, 2, 2), np.ones((1, 1, 1, 2, 3)), "bcxyz", None, RuntimeError],
    ],
)
def test_cvl_read_exc(data_res, read_res, data_read, dim_order, interp_mode, expected_exc, mocker):
    cvl = CVLayer(
        {"cloudpath": "dummy"},
        data_resolution=data_res,
        dim_order=dim_order,
        interpolation_mode=interp_mode,
    )

    bcube = BoundingCube(slices=(slice(1, 1), slice(1, 1), slice(1, 1)))

    idx = (read_res, bcube)

    _read_volume_fn = mocker.patch(
        "zetta_utils.data.layers.volumetric.CVLayer._read_volume",
        return_value=data_read,
    )
    with pytest.raises(expected_exc):
        cvl._read(idx)
