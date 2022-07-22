# pylint: disable=missing-docstring
from typing import Tuple

import pytest

from zetta_utils.types import Vec3D, Coord3D, Slice3D, Padding3D, BoundingCube, Dim3D


@pytest.mark.parametrize(
    "slices, start_coord, end_coord, resolution",
    [
        [
            (slice(0, 1), slice(0, 2), slice(0, 3)),
            (1, 2, 3),
            None,
            (1, 1, 1),
        ],
        [
            (slice(0, 1), slice(0, 2), slice(0, 3)),
            None,
            (1, 2, 3),
            (1, 1, 1),
        ],
        [
            None,
            None,
            (1, 2, 3),
            (1, 1, 1),
        ],
        [
            None,
            (1, 2, 3),
            None,
            (1, 1, 1),
        ],
        [
            None,
            "1, 2, 3",
            "1, 1",
            (1, 1, 1),
        ],
        [
            None,
            "1, 2, 3, 4",
            "1, 1, 1",
            (1, 1, 1),
        ],
    ],
)
def test_constructor_exc(
    slices: Slice3D,
    start_coord: Coord3D,
    end_coord: Coord3D,
    resolution: Vec3D,
):
    with pytest.raises(ValueError):
        BoundingCube(
            slices=slices,
            start_coord=start_coord,
            end_coord=end_coord,
            resolution=resolution,
        )


@pytest.mark.parametrize(
    "slices, start_coord, end_coord, resolution, expected_ranges",
    [
        [
            (slice(0, 1), slice(0, 2), slice(0, 3)),
            None,
            None,
            (1, 1, 1),
            ((0, 1), (0, 2), (0, 3)),
        ],
        [
            (slice(0, 1), slice(0, 2), slice(0, 3)),
            None,
            None,
            (3, 5, 7),
            ((0, 3), (0, 10), (0, 21)),
        ],
        [
            None,
            (1, 2, 3),
            (11, 12, 13),
            (1, 1, 1),
            ((1, 11), (2, 12), (3, 13)),
        ],
        [
            None,
            "1, 2, 3",
            "11, 12, 13",
            (1, 1, 1),
            ((1, 11), (2, 12), (3, 13)),
        ],
    ],
)
def test_constructor(
    slices: Slice3D,
    start_coord: Coord3D,
    end_coord: Coord3D,
    resolution: Vec3D,
    expected_ranges: Tuple[Tuple[int, int], ...],
):
    result = BoundingCube(
        slices=slices,
        start_coord=start_coord,
        end_coord=end_coord,
        resolution=resolution,
    )
    assert result.ranges == expected_ranges


@pytest.fixture
def bcube_get_slices_x0():
    result = BoundingCube(start_coord="0, 0, 0", end_coord="1, 2, 3")
    return result


@pytest.fixture
def bcube_get_slices_x1():
    result = BoundingCube(start_coord="10, 100, 1000", end_coord="20, 110, 1010")
    return result


@pytest.mark.parametrize(
    "bcube_name, resolution, allow_rounding, expected",
    [
        [
            "bcube_get_slices_x0",
            (1, 1, 1),
            False,
            (slice(0, 1), slice(0, 2), slice(0, 3)),
        ],
        [
            "bcube_get_slices_x0",
            (2, 1, 1),
            True,
            (slice(0, 0), slice(0, 2), slice(0, 3)),
        ],
        [
            "bcube_get_slices_x0",
            (2, 2, 2),
            True,
            (slice(0, 0), slice(0, 1), slice(0, 2)),  # round to even
        ],
        [
            "bcube_get_slices_x1",
            (2, 5, 10),
            False,
            (slice(5, 10), slice(20, 22), slice(100, 101)),
        ],
    ],
)
def test_get_slices(
    bcube_name: str, resolution: Vec3D, allow_rounding: bool, expected: Slice3D, request
):
    bcube = request.getfixturevalue(bcube_name)  # type: BoundingCube
    result = bcube.get_slices(resolution=resolution, allow_rounding=allow_rounding)
    assert result == expected


@pytest.mark.parametrize(
    "bcube_name, dim, resolution, allow_rounding",
    [
        [
            "bcube_get_slices_x0",
            0,
            (2, 1, 1),
            False,
        ],
        [
            "bcube_get_slices_x1",
            0,
            (20, 1, 1),
            False,
        ],
    ],
)
def test_get_slice_exc(
    bcube_name: str, dim: Dim3D, resolution: Vec3D, allow_rounding: bool, request
):
    bcube = request.getfixturevalue(bcube_name)  # type: BoundingCube
    with pytest.raises(ValueError):
        bcube.get_slice(dim=dim, resolution=resolution, allow_rounding=allow_rounding)


@pytest.fixture
def bcube_pad_x0():
    result = BoundingCube(start_coord="0, 0, 0", end_coord="0, 0, 0")
    return result


@pytest.fixture
def bcube_pad_x1():
    result = BoundingCube(start_coord="-1, -3, -5", end_coord="1, 3, 5")
    return result


@pytest.fixture
def bcube_pad_x2():
    result = BoundingCube(start_coord="-1, -9, -25", end_coord="1, 9, 25")
    return result


@pytest.fixture
def bcube_pad_x3():
    result = BoundingCube(start_coord="0, -3, -10", end_coord="1, 9, 25")
    return result


@pytest.mark.parametrize(
    "bcube_name, pad, resolution, expected_name",
    [
        [
            "bcube_pad_x0",
            (1, 3, 5),
            (1, 1, 1),
            "bcube_pad_x1",
        ],
        ["bcube_pad_x0", (1, 3, 5), (1, 3, 5), "bcube_pad_x2"],
        ["bcube_pad_x0", ((0, 1), (1, 3), (2, 5)), (1, 3, 5), "bcube_pad_x3"],
    ],
)
def test_pad(
    bcube_name: str, pad: Padding3D, resolution: Vec3D, expected_name: str, request
):
    bcube = request.getfixturevalue(bcube_name)  # type: BoundingCube
    result = bcube.pad(pad=pad, resolution=resolution)
    expected = request.getfixturevalue(expected_name)  # type: BoundingCube
    assert result == expected


@pytest.fixture
def bcube_translate_x0() -> BoundingCube:
    return BoundingCube(start_coord="0, 0, 0", end_coord="1, 1, 1")


@pytest.fixture
def bcube_translate_x1() -> BoundingCube:
    return BoundingCube(start_coord="10, 10, 10", end_coord="11, 11, 11")


@pytest.mark.parametrize(
    "bcube_name, offset, resolution, in_place, expected_name",
    [
        ["bcube_translate_x0", (10, 10, 10), (1, 1, 1), False, "bcube_translate_x1"],
    ],
)
def test_translate(
    bcube_name: str,
    offset: Vec3D,
    resolution: Vec3D,
    in_place: bool,
    expected_name: str,
    request,
):
    bcube = request.getfixturevalue(bcube_name)
    result = bcube.translate(offset, resolution, in_place)
    expected = request.getfixturevalue(expected_name)
    assert result == expected
