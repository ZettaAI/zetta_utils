# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long
import pathlib
import os

import pytest
import numpy as np

from zetta_utils.types import (
    VolumetricIndex,
    BoundingCube,
    CVLayer,
    Vec3D,
    Array,
    InterpolationMode,
    DimOrder3D,
)

THIS_DIR_PATH = pathlib.Path(__file__).parent.resolve()
TEST_DATA_PATH = THIS_DIR_PATH / "data"


def build_fafb_cvl(cv_kwargs: dict = None, **kwargs):
    if cv_kwargs is None:
        cv_kwargs = {}
    cv_kwargs["cloudpath"] = f"file://{TEST_DATA_PATH / 'cvs/fafb_v15_img_norm.cv'}"
    return build_cvl(cv_kwargs, **kwargs)


def build_cvl(
    cv_kwargs,
    readonly: bool = False,
    data_resolution: Vec3D = None,
    index_resolution: Vec3D = None,
    dim_order: DimOrder3D = "cxyz",
    interpolation_mode: InterpolationMode = "img",
):
    return CVLayer(
        cv_params=cv_kwargs,
        index_resolution=index_resolution,
        data_resolution=data_resolution,
        readonly=readonly,
        dim_order=dim_order,
        interpolation_mode=interpolation_mode,
    )


def test_cv_layer_construct():
    build_fafb_cvl()


@pytest.fixture
def fafb_large_2sec_x0():
    result = BoundingCube(
        slices=(
            slice(90 * 1024, 100 * 1024),
            slice(40 * 1024, 50 * 1024),
            slice(2000, 2002),
        ),
        resolution=(4, 4, 40),
    )
    return result


@pytest.fixture
def fafb_large_1sec_x0():
    result = BoundingCube(
        slices=(
            slice(90 * 1024, 100 * 1024),
            slice(40 * 1024, 50 * 1024),
            slice(2000, 2001),
        ),
        resolution=(4, 4, 40),
    )
    return result


@pytest.fixture
def read_fafb_setup(request):
    dim_order = request.node.funcargs["dim_order"]
    volumetric_kwargs = request.node.funcargs["volumetric_kwargs"]
    read_res = request.node.funcargs["read_res"]
    bcube_name = request.node.funcargs["bcube_name"]
    bcube = request.getfixturevalue(bcube_name)
    cvl = build_fafb_cvl(dim_order=dim_order, **volumetric_kwargs)

    data_res = volumetric_kwargs.get("data_resolution", None)
    index_res = volumetric_kwargs.get("index_resolution", None)

    actual_read_res = read_res
    if actual_read_res is None:
        if data_res is not None:
            actual_read_res = data_res
        elif index_res is not None:
            actual_read_res = index_res

    if actual_read_res is None:
        expected = None
    else:
        if data_res is None:
            data_res = actual_read_res

        actual_read_res_str = "-".join((str(i) for i in actual_read_res))
        data_res_str = "-".join((str(i) for i in data_res))

        reference_path = (
            TEST_DATA_PATH
            / f"reference/{bcube_name}_read{actual_read_res_str}_data{data_res_str}_{dim_order}.npy"
        )
        if os.path.exists(reference_path):
            expected = np.load(reference_path)
        else:
            expected = None

    idx = (read_res, bcube)
    return {
        "cvl": cvl,
        "idx": idx,
        "expected": expected,
    }


@pytest.mark.parametrize(
    "fixture_name, bcube_name, dim_order, read_res, volumetric_kwargs",
    [
        ["read_fafb_setup", "fafb_large_2sec_x0", "cxyz", (64, 64, 40), {}],
        [
            "read_fafb_setup",
            "fafb_large_2sec_x0",
            "cxyz",
            (64, 64, 40),
            {"data_resolution": (64, 64, 40)},
        ],
        [
            "read_fafb_setup",
            "fafb_large_2sec_x0",
            "cxyz",
            None,
            {"data_resolution": (64, 64, 40)},
        ],
        [
            "read_fafb_setup",
            "fafb_large_2sec_x0",
            "cxyz",
            None,
            {"index_resolution": (64, 64, 40)},
        ],
        [
            "read_fafb_setup",
            "fafb_large_2sec_x0",
            "bcxyz",
            (64, 64, 40),
            {"data_resolution": (128, 128, 40)},
        ],
        [
            "read_fafb_setup",
            "fafb_large_2sec_x0",
            "bcxyz",
            (64, 64, 40),
            {"data_resolution": (32, 32, 40)},
        ],
        [
            "read_fafb_setup",
            "fafb_large_1sec_x0",
            "bcxyz",
            (64, 64, 40),
            {"data_resolution": (128, 128, 40)},
        ],
        [
            "read_fafb_setup",
            "fafb_large_1sec_x0",
            "bcxy",
            (64, 64, 40),
            {"data_resolution": (128, 128, 40)},
        ],
        [
            "read_fafb_setup",
            "fafb_large_2sec_x0",
            "bcxyz",
            (32, 128, 20),
            {"data_resolution": (64, 64, 40)},
        ],
        [
            "read_fafb_setup",
            "fafb_large_2sec_x0",
            "bcxyz",
            (64, 64, 20),
            {"data_resolution": (64, 64, 40)},
        ],
    ],
)
def test_cv_layer_read(fixture_name, bcube_name, dim_order, read_res, volumetric_kwargs, request):
    fixture_val = request.getfixturevalue(fixture_name)
    cvl = fixture_val["cvl"]  # type: CVLayer
    idx = fixture_val["idx"]  # type: VolumetricIndex
    expected = fixture_val["expected"]  # type: Array

    result = cvl[idx]
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "fixture_name, bcube_name, dim_order, read_res, volumetric_kwargs, expected_exc",
    [
        ["read_fafb_setup", "fafb_large_2sec_x0", "cxyz", None, {}, RuntimeError],
        [
            "read_fafb_setup",
            "fafb_large_2sec_x0",
            "cxyz",
            (64, 64, 40),
            {"data_resolution": (128, 128, 40), "interpolation_mode": None},
            RuntimeError,
        ],
        [
            "read_fafb_setup",
            "fafb_large_1sec_x0",
            "bcxyz",
            (64, 64, 30),
            {"data_resolution": (64, 64, 40)},
            RuntimeError,
        ],
        [
            "read_fafb_setup",
            "fafb_large_2sec_x0",
            "cxyz",
            (64, 64, 40),
            {"data_resolution": (128, 128, 40)},
            RuntimeError,
        ],
        [
            "read_fafb_setup",
            "fafb_large_2sec_x0",
            "bcxy",
            (64, 64, 40),
            {},
            RuntimeError,
        ],
    ],
)
def test_cv_layer_read_exc(
    fixture_name,
    bcube_name,
    dim_order,
    read_res,
    volumetric_kwargs,
    expected_exc,
    request,
):
    fixture_val = request.getfixturevalue(fixture_name)
    cvl = fixture_val["cvl"]  # type: CVLayer
    idx = fixture_val["idx"]  # type: VolumetricIndex

    with pytest.raises(expected_exc):
        cvl[idx]
