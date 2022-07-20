# pylint: disable=missing-docstring
import pathlib
from typing import List, Union

import pytest
import numpy as np

import zetta_utils as zu

from zetta_utils.data.layers.volumetric import (
    VolumetricIndex,
    CVLayer,
    _convert_to_vol_idx,
)
from zetta_utils.bcube import BoundingCube

THIS_DIR_PATH = pathlib.Path(__file__).parent.resolve()
TEST_DATA_PATH = THIS_DIR_PATH / "data"


# @pytest.fixture(params=['dir1_fixture', 'dir2_fixture'])
# def dirname(request):
#        return request.getfixturevalue(request.param)

SLICES_X0 = [slice(0, 10), slice(0, 100), slice(0, 100)]
BCUBE_X0 = BoundingCube(slices=SLICES_X0)


@pytest.mark.parametrize(
    "in_idx, index_resolution, expected",
    [
        [[BCUBE_X0], None, VolumetricIndex(resolution=None, bcube=BCUBE_X0)],
        [
            [[4, 4, 4], BCUBE_X0],
            None,
            VolumetricIndex(resolution=[4, 4, 4], bcube=BCUBE_X0),
        ],
        [[[4, 4, 4]] + SLICES_X0, [1, 1, 1], VolumetricIndex(resolution=[4, 4, 4], bcube=BCUBE_X0)],  # type: ignore # pylint: disable=line-too-long
        [SLICES_X0, [1, 1, 1], VolumetricIndex(resolution=None, bcube=BCUBE_X0)],
        [
            [slice(0, 10), slice(0, 50), slice(0, 50)],
            [1, 2, 2],
            VolumetricIndex(resolution=None, bcube=BCUBE_X0),
        ],
    ],
)
def test_convert_to_vol_idx(
    in_idx: Union[VolumetricIndex, list],
    index_resolution: List[int],
    expected: VolumetricIndex,
):
    result = _convert_to_vol_idx(in_idx, index_resolution)
    assert result == expected


def build_fafb_cvl(cv_kwargs: dict = None, **kwargs):
    if cv_kwargs is None:
        cv_kwargs = {}
    cv_kwargs["cloudpath"] = f"file://{TEST_DATA_PATH / 'cvs/fafb_v15_img_norm.cv'}"
    return build_cvl(cv_kwargs, **kwargs)


def build_cvl(
    cv_kwargs,
    readonly: bool = False,
    data_resolution: List[int] = None,
    index_resolution: List[int] = None,
    dim_order: zu.types.DimOrder = "xyzc",
):
    if index_resolution is None:
        index_resolution = [4, 4, 40]
    return CVLayer(
        cv_params=cv_kwargs,
        index_resolution=index_resolution,
        data_resolution=data_resolution,
        readonly=readonly,
        dim_order=dim_order,
    )


def test_cv_layer_construct():
    build_fafb_cvl()


SLICES_X1 = [slice(90000, 100000), slice(40000, 50000), slice(2000, 2002)]
BCUBE_X1 = BoundingCube(slices=SLICES_X1, resolution=[4, 4, 40])


@pytest.mark.parametrize(
    "cvl, idx, reference_npy_path",
    [
        [
            build_fafb_cvl(dim_order="xyzc"),
            VolumetricIndex(resolution=[64, 64, 40], bcube=BCUBE_X1),
            TEST_DATA_PATH / "reference/fafb_64nm_x1_xyzc.npy",
        ],
        [
            build_fafb_cvl(dim_order="cxyz"),
            VolumetricIndex(resolution=[64, 64, 40], bcube=BCUBE_X1),
            TEST_DATA_PATH / "reference/fafb_64nm_x1_cxyz.npy",
        ],
    ],
)
def test_cv_layer_read(cvl: CVLayer, idx: VolumetricIndex, reference_npy_path: str):
    result = cvl[idx]
    expected = np.load(reference_npy_path)
    # import pdb; pdb.set_trace()
    np.testing.assert_array_equal(result, expected)
