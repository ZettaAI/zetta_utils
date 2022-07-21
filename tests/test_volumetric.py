# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement
import pathlib

import pytest
import numpy as np

import zetta_utils as zu

from zetta_utils.data.layers.volumetric import _standardize_vol_idx
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


@pytest.fixture
def res_slices_x0():
    return ((4, 4, 4), slice(0, 10), slice(0, 100), slice(0, 100))


@pytest.fixture
def slices_x0():
    return (slice(0, 10), slice(0, 100), slice(0, 100))


@pytest.fixture
def bcube_x0(slices_x0):
    return BoundingCube(slices=slices_x0)


@pytest.fixture
def standardize_idx_bcube(bcube_x0):
    return [(bcube_x0), None, (None, bcube_x0)]


@pytest.fixture
def standardize_idx_res_bcube(bcube_x0):
    return [
        ((4, 4, 4), bcube_x0),
        None,
        ((4, 4, 4), bcube_x0),
    ]


@pytest.fixture
def standardize_idx_res_slices(res_slices_x0, bcube_x0):
    return [res_slices_x0, (1, 1, 1), ((4, 4, 4), bcube_x0)]


@pytest.fixture
def standardize_idx_slices(slices_x0, bcube_x0):
    return [slices_x0, (1, 1, 1), (None, bcube_x0)]


@pytest.fixture
def standardize_idx_slices_nonu_range(bcube_x0):
    return [
        (slice(0, 10), slice(0, 50), slice(0, 50)),
        (1, 2, 2),
        (None, bcube_x0),
    ]


@pytest.mark.parametrize(
    "test_fixture_name",
    [
        "standardize_idx_bcube",
        "standardize_idx_res_bcube",
        "standardize_idx_res_slices",
        "standardize_idx_slices",
        "standardize_idx_slices_nonu_range",
    ],
)
def test_standardize_idx(
    test_fixture_name: str,
    request,
):
    test_fixture = request.getfixturevalue(test_fixture_name)
    in_idx = test_fixture[0]  # type: VolumetricIndex
    index_resolution = test_fixture[1]  # type: Vec3D
    expected = test_fixture[2]  # type: VolumetricIndex

    result = _standardize_vol_idx(in_idx, index_resolution)
    assert result == expected


@pytest.fixture
def malformed_index_x0(slices_x0):
    return [(None,) + slices_x0, None]


@pytest.mark.parametrize(
    "test_fixture_name",
    [
        "malformed_index_x0",
    ],
)
def test_standardize_idx_exc(
    test_fixture_name: str,
    request,
):
    test_fixture = request.getfixturevalue(test_fixture_name)
    in_idx = test_fixture[0]  # type: VolumetricIndex
    index_resolution = test_fixture[1]  # type: Vec3D
    with pytest.raises(ValueError):
        _standardize_vol_idx(in_idx, index_resolution)


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
def read_fafb_setup(request):
    dim_order = request.node.funcargs["dim_order"]
    volumetric_kwargs = request.node.funcargs["volumetric_kwargs"]
    read_res = request.node.funcargs["read_res"]
    cvl = build_fafb_cvl(dim_order=dim_order, **volumetric_kwargs)
    bcube = BoundingCube(
        slices=(
            slice(90000, 100000),
            slice(40000, 50000),
            slice(2000, 2002),
        ),
        resolution=(4, 4, 40),
    )
    if read_res is None:
        idx = bcube
    else:
        idx = (read_res, bcube)

    reference_path = TEST_DATA_PATH / f"reference/fafb_64nm_x1_{dim_order}.npy"
    expected = np.load(reference_path)

    return {
        "cvl": cvl,
        "idx": idx,
        "expected": expected,
    }


@pytest.mark.parametrize(
    "fixture_name, dim_order, read_res, volumetric_kwargs",
    [
        # ['read_fafb_setup', 'xyzc', (64, 64, 40), {}],
        ["read_fafb_setup", "cxyz", (64, 64, 40), {}],
        ["read_fafb_setup", "cxyz", (64, 64, 40), {"data_resolution": (64, 64, 40)}],
        ["read_fafb_setup", "cxyz", None, {"data_resolution": (64, 64, 40)}],
        ["read_fafb_setup", "cxyz", None, {"index_resolution": (64, 64, 40)}],
        ["read_fafb_setup", "cxyz", (64, 64, 40), {"data_resolution": (128, 128, 40)}],
    ],
)
def test_cv_layer_read(fixture_name, dim_order, read_res, volumetric_kwargs, request):
    fixture_val = request.getfixturevalue(fixture_name)
    cvl = fixture_val["cvl"]  # type: CVLayer
    idx = fixture_val["idx"]  # type: VolumetricIndex
    expected = fixture_val["expected"]  # type: Array

    result = cvl[idx]
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "fixture_name, dim_order, read_res, volumetric_kwargs",
    [
        [
            "read_fafb_setup",
            "cxyz",
            (64, 64, 40),
            {"data_resolution": (128, 128, 40), "interpolation_mode": None},
        ],
        ["read_fafb_setup", "cxyz", None, {}],
    ],
)
def test_cv_layer_read_exc(
    fixture_name, dim_order, read_res, volumetric_kwargs, request
):
    fixture_val = request.getfixturevalue(fixture_name)
    cvl = fixture_val["cvl"]  # type: CVLayer
    idx = fixture_val["idx"]  # type: VolumetricIndex

    with pytest.raises(ValueError):
        cvl[idx]


@pytest.fixture
def translation_setup_x0(request):
    idx_type = request.node.funcargs["idx_type"]
    offset_resolution = request.node.funcargs["offset_resolution"]
    slices = (slice(0, 1), slice(0, 2), slice(0, 3))

    offset = (10, 20, 30)
    if offset_resolution is None:
        expected_slices = (slice(10, 11), slice(20, 22), slice(30, 33))
    elif offset_resolution == (2, 2, 2):
        expected_slices = (slice(20, 21), slice(40, 42), slice(60, 63))
    else:
        raise ValueError(f"Unsupported test offset_resolutoin: {offset_resolution}")

    if idx_type == "[bcube]":
        idx = BoundingCube(slices=slices)
        expected = BoundingCube(slices=expected_slices)
    elif idx_type == "[resolution, bcube]":
        idx = (
            None,
            BoundingCube(slices=slices),
        )
        expected = (
            None,
            BoundingCube(slices=expected_slices),
        )
    elif idx_type == "[slices]":
        idx = slices
        expected = expected_slices
    elif idx_type == "[resolution, *slices]":
        idx = (None,) + slices
        expected = (None,) + expected_slices
    return {
        "idx": idx,
        "expected": expected,
        "offset": offset,
    }


@pytest.mark.parametrize(
    "fixture_name, idx_type, offset_resolution",
    [
        ["translation_setup_x0", "[bcube]", None],
        ["translation_setup_x0", "[bcube]", (2, 2, 2)],
        ["translation_setup_x0", "[resolution, bcube]", None],
        ["translation_setup_x0", "[resolution, bcube]", (2, 2, 2)],
        ["translation_setup_x0", "[slices]", None],
        ["translation_setup_x0", "[resolution, *slices]", None],
    ],
)
def test_translate_vol_idx(fixture_name, idx_type, offset_resolution, request):
    fixture_val = request.getfixturevalue(fixture_name)
    idx = fixture_val["idx"]  # type: VolumetricIndex
    offset = fixture_val["offset"]  # type: Vec3D
    expected = fixture_val["expected"]  # type: Array

    result = zu.data.layers.volumetric.translate_volumetric_index(
        idx, offset, offset_resolution
    )

    assert result == expected


@pytest.mark.parametrize(
    "fixture_name, idx_type, offset_resolution",
    [
        ["translation_setup_x0", "[slices]", (2, 2, 2)],
        ["translation_setup_x0", "[resolution, *slices]", (2, 2, 2)],
    ],
)
def test_translate_vol_idx_exc(fixture_name, idx_type, offset_resolution, request):
    fixture_val = request.getfixturevalue(fixture_name)
    idx = fixture_val["idx"]  # type: VolumetricIndex
    offset = fixture_val["offset"]  # type: Vec3D

    with pytest.raises(ValueError):
        zu.data.layers.volumetric.translate_volumetric_index(
            idx, offset, offset_resolution
        )
