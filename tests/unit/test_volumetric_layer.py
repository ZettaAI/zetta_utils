# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long
import pytest

import zetta_utils as zu

from zetta_utils.data.layers.volumetric import _standardize_vol_idx
from zetta_utils.types import (
    VolumetricIndex,
    BoundingCube,
    Vec3D,
    Array,
)


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
