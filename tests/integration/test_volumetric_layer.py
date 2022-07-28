# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long
import pathlib
import os
from typing import Optional

import pytest
import numpy as np

from zetta_utils.bbox import BoundingCube
from zetta_utils.tensor.ops import InterpolationMode
from zetta_utils.io.layers import cv_layer

from zetta_utils.typing import (
    Vec3D,
    Tensor,
)

THIS_DIR_PATH = pathlib.Path(__file__).parent.resolve()
TEST_DATA_PATH = THIS_DIR_PATH / "assets"


def build_cvl(
    path,
    cv_kwargs,
    readonly: bool = False,
    data_resolution: Optional[Vec3D] = None,
    index_resolution: Optional[Vec3D] = None,
    default_desired_resolution: Optional[Vec3D] = None,
    interpolation_mode: InterpolationMode = "img",
):
    return cv_layer(
        path=path,
        cv_params=cv_kwargs,
        index_resolution=index_resolution,
        default_desired_resolution=default_desired_resolution,
        data_resolution=data_resolution,
        readonly=readonly,
        interpolation_mode=interpolation_mode,
    )


def build_fafb_cvl(cv_kwargs: Optional[dict] = None, **kwargs):
    if cv_kwargs is None:
        cv_kwargs = {}
    return build_cvl(f"file://{TEST_DATA_PATH / 'cvs/fafb_v15_img_norm.cv'}", cv_kwargs, **kwargs)


def test_cv_layer_construct():
    build_fafb_cvl()


@pytest.fixture
def fafb_large_2sec_x0():
    result = BoundingCube.from_slices(
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
    result = BoundingCube.from_slices(
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
    volumetric_kwargs = request.node.funcargs["volumetric_kwargs"]
    read_res = request.node.funcargs["read_res"]
    bcube_name = request.node.funcargs["bcube_name"]
    bcube = request.getfixturevalue(bcube_name)
    cvl = build_fafb_cvl(**volumetric_kwargs)

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
            / f"reference/{bcube_name}_read{actual_read_res_str}_data{data_res_str}_bcxyz.npy"
        )
        if os.path.exists(reference_path):
            expected = np.load(reference_path)
        else:
            expected = None

    idx = (read_res,) + (bcube.to_slices(read_res))
    return {
        "cvl": cvl,
        "idx": idx,
        "expected": expected,
    }


@pytest.mark.parametrize(
    "fixture_name, bcube_name, read_res, volumetric_kwargs",
    [
        [
            "read_fafb_setup",
            "fafb_large_2sec_x0",
            (64, 64, 40),
            {"data_resolution": (128, 128, 40)},
        ],
        [
            "read_fafb_setup",
            "fafb_large_2sec_x0",
            (64, 64, 40),
            {"data_resolution": (32, 32, 40)},
        ],
        [
            "read_fafb_setup",
            "fafb_large_1sec_x0",
            (64, 64, 40),
            {"data_resolution": (128, 128, 40)},
        ],
        [
            "read_fafb_setup",
            "fafb_large_2sec_x0",
            (32, 128, 20),
            {"data_resolution": (64, 64, 40)},
        ],
        [
            "read_fafb_setup",
            "fafb_large_2sec_x0",
            (64, 64, 20),
            {"data_resolution": (64, 64, 40)},
        ],
    ],
)
def test_cv_layer_read(fixture_name, bcube_name, read_res, volumetric_kwargs, request):
    fixture_val = request.getfixturevalue(fixture_name)
    cvl = fixture_val["cvl"]  # type: cv_layer
    idx = fixture_val["idx"]  # type: VolumetricIndex
    expected = fixture_val["expected"]  # type: Tensor
    result = cvl[idx]
    np.testing.assert_array_equal(result, expected)
