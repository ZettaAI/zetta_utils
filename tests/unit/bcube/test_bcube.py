# pylint: disable=missing-docstring
import pytest

from zetta_utils.bcube import BoundingCube
from zetta_utils.typing import Slices3D, Vec3D


@pytest.mark.parametrize(
    "slices, resolution",
    [
        [
            (slice(0, 1), slice(0, 2), slice(0, 3)),
            (1, 1, 1, 4),
        ],
        [
            (slice(0, 1, 1), slice(0, 2), slice(0, 3)),
            (1, 1, 1),
        ],
    ],
)
def test_from_slices_exc(
    slices: Slices3D,
    resolution: Vec3D,
):
    with pytest.raises(ValueError):
        BoundingCube.from_slices(
            slices=slices,
            resolution=resolution,
        )


@pytest.mark.parametrize(
    "start_coord, end_coord, resolution",
    [
        [
            (1, 2, 3),
            (1, 2, 3, 4),
            (1, 1, 1),
        ],
    ],
)
def test_from_coords_exc(
    start_coord: Vec3D,
    end_coord: Vec3D,
    resolution: Vec3D,
):
    with pytest.raises(ValueError):
        BoundingCube.from_coords(
            start_coord=start_coord,
            end_coord=end_coord,
            resolution=resolution,
        )


@pytest.mark.parametrize(
    "slices, resolution, expected_bounds",
    [
        [
            (slice(0, 1), slice(0, 2), slice(0, 3)),
            (1, 1, 1),
            ((0, 1), (0, 2), (0, 3)),
        ],
        [
            (slice(0, 1), slice(0, 2), slice(0, 3)),
            (3, 5, 7),
            ((0, 3), (0, 10), (0, 21)),
        ],
    ],
)
def test_from_slices(slices: Slices3D, resolution: Vec3D, expected_bounds: Slices3D):
    result = BoundingCube.from_slices(
        slices=slices,
        resolution=resolution,
    )
    assert result.bounds == expected_bounds


@pytest.mark.parametrize(
    "start_coord, end_coord, resolution, expected_bounds",
    [
        [
            (1, 2, 3),
            (11, 12, 13),
            (1, 1, 1),
            ((1, 11), (2, 12), (3, 13)),
        ],
        [
            (1, 2, 3),
            (11, 12, 13),
            (2, 4, 8),
            ((2, 22), (8, 48), (24, 104)),
        ],
    ],
)
def test_constructor(
    start_coord: Vec3D, end_coord: Vec3D, resolution: Vec3D, expected_bounds: Slices3D
):
    result = BoundingCube.from_coords(
        start_coord=start_coord,
        end_coord=end_coord,
        resolution=resolution,
    )
    assert result.bounds == expected_bounds


@pytest.mark.parametrize(
    "bcube, resolution, allow_slice_rounding, expected",
    [
        [
            BoundingCube(bounds=((0, 1), (0, 2), (0, 3))),
            (1, 1, 1),
            False,
            (slice(0, 1), slice(0, 2), slice(0, 3)),
        ],
        [
            BoundingCube(bounds=((0, 1), (0, 2), (0, 3))),
            (2, 1, 1),
            True,
            (slice(0, 0), slice(0, 2), slice(0, 3)),
        ],
        [
            BoundingCube(bounds=((0, 1), (0, 2), (0, 3))),
            (2, 2, 2),
            True,
            (slice(0, 0), slice(0, 1), slice(0, 2)),  # round to even
        ],
        [
            BoundingCube(bounds=((10, 20), (100, 110), (1000, 1010))),
            (2, 5, 10),
            False,
            (slice(5, 10), slice(20, 22), slice(100, 101)),
        ],
    ],
)
def test_to_slices(
    bcube: BoundingCube, resolution: Vec3D, allow_slice_rounding: bool, expected: Slices3D
):
    result = bcube.to_slices(resolution=resolution, allow_slice_rounding=allow_slice_rounding)
    assert result == expected


@pytest.mark.parametrize(
    "bcube, dim, resolution, allow_slice_rounding",
    [
        [
            BoundingCube(bounds=((0, 1), (0, 2), (0, 3))),
            0,
            (2, 1, 1),
            False,
        ],
        [
            BoundingCube(bounds=((10, 20), (100, 110), (1000, 1010))),
            0,
            (20, 1, 1),
            False,
        ],
    ],
)
def test_get_slice_exc(
    bcube: BoundingCube,
    dim: int,
    resolution: Vec3D,
    allow_slice_rounding: bool,
):
    with pytest.raises(ValueError):
        bcube.get_slice(dim=dim, resolution=resolution, allow_slice_rounding=allow_slice_rounding)


@pytest.mark.parametrize(
    "bcube, pad, resolution, expected",
    [
        [
            BoundingCube(bounds=((0, 0), (0, 0), (0, 0))),
            (1, 3, 5),
            (1, 1, 1),
            BoundingCube(bounds=((-1, 1), (-3, 3), (-5, 5))),
        ],
        [
            BoundingCube(bounds=((0, 0), (0, 0), (0, 0))),
            (1, 3, 5),
            (1, 3, 5),
            BoundingCube(bounds=((-1, 1), (-9, 9), (-25, 25))),
        ],
        [
            BoundingCube(bounds=((0, 0), (0, 0), (0, 0))),
            ((0, 1), (1, 3), (2, 5)),
            (1, 3, 5),
            BoundingCube(bounds=((0, 1), (-3, 9), (-10, 25))),
        ],
    ],
)
def test_pad(bcube: BoundingCube, pad, resolution: Vec3D, expected: BoundingCube):
    result = bcube.pad(pad=pad, resolution=resolution)
    assert result == expected


@pytest.mark.parametrize(
    "bcube, pad, resolution, expected_exc",
    [
        [
            BoundingCube(bounds=((0, 0), (0, 0), (0, 0))),
            (1, 3, 5, 3),
            (1, 1, 1),
            ValueError,
        ],
    ],
)
def test_pad_exc(bcube: BoundingCube, pad, resolution: Vec3D, expected_exc):
    with pytest.raises(expected_exc):
        bcube.pad(pad=pad, resolution=resolution)


@pytest.mark.parametrize(
    "bcube, offset, resolution, in_place, expected",
    [
        [
            BoundingCube(bounds=((0, 1), (0, 1), (0, 1))),
            (10, 10, 10),
            (1, 1, 1),
            False,
            BoundingCube(bounds=((10, 11), (10, 11), (10, 11))),
        ],
    ],
)
def test_translate(
    bcube: BoundingCube,
    offset: Vec3D,
    resolution: Vec3D,
    in_place: bool,
    expected: BoundingCube,
):
    result = bcube.translate(offset, resolution, in_place)
    assert result == expected


@pytest.mark.parametrize(
    "bcube, offset, resolution, in_place, expected",
    [
        [
            BoundingCube(bounds=((0, 10), (0, 10), (0, 10))),
            (2, 2, 2),
            (1, 1, 1),
            False,
            BoundingCube(bounds=((2, 10), (2, 10), (2, 10))),
        ],
    ],
)
def test_translate_start(
    bcube: BoundingCube,
    offset: Vec3D,
    resolution: Vec3D,
    in_place: bool,
    expected: BoundingCube,
):
    result = bcube.translate_start(offset, resolution, in_place)
    assert result == expected


@pytest.mark.parametrize(
    "bcube, offset, resolution, in_place, expected",
    [
        [
            BoundingCube(bounds=((0, 10), (0, 10), (0, 10))),
            (2, 2, 2),
            (1, 1, 1),
            False,
            BoundingCube(bounds=((0, 12), (0, 12), (0, 12))),
        ],
    ],
)
def test_translate_stop(
    bcube: BoundingCube,
    offset: Vec3D,
    resolution: Vec3D,
    in_place: bool,
    expected: BoundingCube,
):
    result = bcube.translate_stop(offset, resolution, in_place)
    assert result == expected


@pytest.mark.parametrize(
    "bcube, crop, resolution, expected",
    [
        [
            BoundingCube(bounds=((-1, 1), (-3, 3), (-5, 5))),
            (1, 3, 5),
            (1, 1, 1),
            BoundingCube(bounds=((0, 0), (0, 0), (0, 0))),
        ],
        [
            BoundingCube(bounds=((-1, 1), (-9, 9), (-25, 25))),
            (1, 3, 5),
            (1, 3, 5),
            BoundingCube(bounds=((0, 0), (0, 0), (0, 0))),
        ],
        [
            BoundingCube(bounds=((0, 1), (-3, 9), (-10, 25))),
            ((0, 1), (1, 3), (2, 5)),
            (1, 3, 5),
            BoundingCube(bounds=((0, 0), (0, 0), (0, 0))),
        ],
    ],
)
def test_crop(bcube: BoundingCube, crop, resolution: Vec3D, expected: BoundingCube):
    result = bcube.crop(crop=crop, resolution=resolution)
    assert result == expected


@pytest.mark.parametrize(
    "bcube, crop, resolution, expected_exc",
    [
        [
            BoundingCube(bounds=((0, 0), (0, 0), (0, 0))),
            (1, 3, 5, 3),
            (1, 1, 1),
            ValueError,
        ],
    ],
)
def test_crop_exc(bcube: BoundingCube, crop, resolution: Vec3D, expected_exc):
    with pytest.raises(expected_exc):
        bcube.crop(crop=crop, resolution=resolution)
