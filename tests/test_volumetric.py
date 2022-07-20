# pylint: disable=missing-docstring
import pathlib

import pytest
import numpy as np

import zetta_utils as zu

from zetta_utils.data.layers.volumetric import (
    VolumetricIndex,
    CVLayer,
    _standardize_vol_idx,
)
from zetta_utils.bcube import BoundingCube

THIS_DIR_PATH = pathlib.Path(__file__).parent.resolve()
TEST_DATA_PATH = THIS_DIR_PATH / "data"


# @pytest.fixture(params=['dir1_fixture', 'dir2_fixture'])
# def dirname(request):
#        return request.getfixturevalue(request.param)

SLICES_X0 = (slice(0, 10), slice(0, 100), slice(0, 100))
RES_SLICES_X0 = ((4, 4, 4), slice(0, 10), slice(0, 100), slice(0, 100))
BCUBE_X0 = BoundingCube(slices=SLICES_X0)


@pytest.mark.parametrize(
    "in_idx, index_resolution, expected",
    [
        [(BCUBE_X0), None, (None, BCUBE_X0)],
        [
            ((4, 4, 4), BCUBE_X0),
            None,
            ((4, 4, 4), BCUBE_X0),
        ],
        [RES_SLICES_X0, (1, 1, 1), ((4, 4, 4), BCUBE_X0)],  # type: ignore
        [SLICES_X0, (1, 1, 1), (None, BCUBE_X0)],
        [
            (slice(0, 10), slice(0, 50), slice(0, 50)),
            (1, 2, 2),
            (None, BCUBE_X0),
        ],
    ],
)
def test_standardize_idx(
    in_idx: VolumetricIndex,
    index_resolution: zu.types.VolumetricResolution,
    expected: VolumetricIndex,
):
    result = _standardize_vol_idx(in_idx, index_resolution)
    assert result == expected


def build_fafb_cvl(cv_kwargs: dict = None, **kwargs):
    if cv_kwargs is None:
        cv_kwargs = {}
    cv_kwargs["cloudpath"] = f"file://{TEST_DATA_PATH / 'cvs/fafb_v15_img_norm.cv'}"
    return build_cvl(cv_kwargs, **kwargs)


def build_cvl(
    cv_kwargs,
    readonly: bool = False,
    data_resolution: zu.types.VolumetricResolution = None,
    index_resolution: zu.types.VolumetricResolution = None,
    dim_order: zu.types.VolumetricDimOrder = "xyzc",
):
    if index_resolution is None:
        index_resolution = (4, 4, 40)
    return CVLayer(
        cv_params=cv_kwargs,
        index_resolution=index_resolution,
        data_resolution=data_resolution,
        readonly=readonly,
        dim_order=dim_order,
    )


def test_cv_layer_construct():
    build_fafb_cvl()


SLICES_X1 = (slice(90000, 100000), slice(40000, 50000), slice(2000, 2002))
BCUBE_X1 = BoundingCube(slices=SLICES_X1, resolution=(4, 4, 40))


@pytest.mark.parametrize(
    "cvl, idx, reference_npy_path",
    [
        [
            build_fafb_cvl(dim_order="xyzc"),
            ((64, 64, 40), BCUBE_X1),
            TEST_DATA_PATH / "reference/fafb_64nm_x1_xyzc.npy",
        ],
        [
            build_fafb_cvl(dim_order="cxyz"),
            ((64, 64, 40), BCUBE_X1),
            TEST_DATA_PATH / "reference/fafb_64nm_x1_cxyz.npy",
        ],
    ],
)
def test_cv_layer_read(cvl: CVLayer, idx: VolumetricIndex, reference_npy_path: str):
    result = cvl[idx]
    expected = np.load(reference_npy_path)
    np.testing.assert_array_equal(result, expected)
