# pylint: disable=missing-docstring
import pathlib
import pytest
import numpy as np

from zetta_utils.data_layers.volumetric import (
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
    "in_idx, index_xy_res, expected",
    [
        [[BCUBE_X0], None, VolumetricIndex(xy_res=None, bcube=BCUBE_X0)],
        [[100, BCUBE_X0], None, VolumetricIndex(xy_res=100, bcube=BCUBE_X0)],
        [[100] + SLICES_X0, 1, VolumetricIndex(xy_res=100, bcube=BCUBE_X0)],  # type: ignore # pylint: disable=line-too-long
        [SLICES_X0, 1, VolumetricIndex(xy_res=None, bcube=BCUBE_X0)],
        [
            [slice(0, 10), slice(0, 50), slice(0, 50)],
            2,
            VolumetricIndex(xy_res=None, bcube=BCUBE_X0),
        ],
    ],
)
def test_convert_to_vol_idx(in_idx, index_xy_res, expected):
    result = _convert_to_vol_idx(in_idx, index_xy_res)
    assert result == expected


def build_fafb_cvl(cv_kwargs=None, **kwargs):
    if cv_kwargs is None:
        cv_kwargs = {}
    cv_kwargs["cloudpath"] = f"file://{TEST_DATA_PATH / 'cvs/fafb_v15_img_norm.cv'}"
    return build_cvl(cv_kwargs, **kwargs)


def build_cvl(cv_kwargs, readonly=False, data_xy_res=None, index_xy_res=4):
    return CVLayer(
        cv_params=cv_kwargs,
        z_res=40,
        index_xy_res=index_xy_res,
        data_xy_res=data_xy_res,
        readonly=readonly,
    )


def test_cv_layer_construct():
    build_fafb_cvl()


SLICES_X1 = [slice(2000, 2002), slice(40000, 50000), slice(90000, 100000)]
BCUBE_X1 = BoundingCube(slices=SLICES_X1, xy_res=4)


@pytest.mark.parametrize(
    "cvl, idx, reference_npy_path",
    [
        [
            build_fafb_cvl(),
            VolumetricIndex(xy_res=64, bcube=BCUBE_X1),
            TEST_DATA_PATH / "reference/fafb_64nm_x1.npy",
        ]
    ],
)
def test_cv_layer_read(cvl, idx, reference_npy_path):
    result = cvl[idx]
    expected = np.load(reference_npy_path)
    # import pdb; pdb.set_trace()
    np.testing.assert_array_equal(result, expected)
