# pylint: disable=missing-docstring
from typing import Sequence

import pytest

from zetta_utils.geometry import BBox3D, Vec3D

Slices3D = tuple[slice, slice, slice]


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
        BBox3D.from_slices(
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
    with pytest.raises(Exception):
        BBox3D.from_coords(
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
    result = BBox3D.from_slices(
        slices=slices,
        resolution=resolution,
    )
    assert result.bounds == expected_bounds


@pytest.mark.parametrize(
    "points, resolution, expected_bounds",
    [
        [
            [(1, 2, 5), (4, 6, 3), (11, 8, 8), (5, 12, 13)],
            (1, 1, 1),
            ((1, 11.0001), (2, 12.0001), (3, 13.0001)),
        ],
    ],
)
def test_from_points(points: Sequence[Vec3D], resolution: Vec3D, expected_bounds: Slices3D):
    result = BBox3D.from_points(points=points, resolution=resolution)
    assert result.bounds == expected_bounds


def test_from_points_exc():
    with pytest.raises(ValueError):
        BBox3D.from_points([])
    with pytest.raises(ValueError):
        BBox3D.from_points([(1, 2)])
    with pytest.raises(ValueError):
        BBox3D.from_points([(1, 2, 3, 4)])
    with pytest.raises(ValueError):
        BBox3D.from_points([(1, 2, 3)], resolution=(1, 1))


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
    result = BBox3D.from_coords(
        start_coord=start_coord,
        end_coord=end_coord,
        resolution=resolution,
    )
    assert result.bounds == expected_bounds


@pytest.mark.parametrize(
    "bbox, resolution, allow_slice_rounding, expected",
    [
        [
            BBox3D(bounds=((0, 1), (0, 2), (0, 3))),
            (1, 1, 1),
            False,
            (slice(0, 1), slice(0, 2), slice(0, 3)),
        ],
        [
            BBox3D(bounds=((0, 1), (0, 2), (0, 3))),
            (2, 1, 1),
            True,
            (slice(0, 0), slice(0, 2), slice(0, 3)),
        ],
        [
            BBox3D(bounds=((0, 1), (0, 2), (0, 3))),
            (2, 2, 2),
            True,
            (slice(0, 0), slice(0, 1), slice(0, 2)),  # round to even
        ],
        [
            BBox3D(bounds=((10, 20), (100, 110), (1000, 1010))),
            (2, 5, 10),
            False,
            (slice(5, 10), slice(20, 22), slice(100, 101)),
        ],
    ],
)
def test_to_slices(
    bbox: BBox3D, resolution: Vec3D, allow_slice_rounding: bool, expected: Slices3D
):
    result = bbox.to_slices(resolution=resolution, allow_slice_rounding=allow_slice_rounding)
    assert result == expected


def test_to_slices_exc():
    bbox = BBox3D(bounds=((10, 20), (100, 110), (1000, 1010)))
    with pytest.raises(ValueError):
        bbox.to_slices(resolution=(1, 2))


@pytest.mark.parametrize(
    "bbox, dim, resolution, allow_slice_rounding",
    [
        [
            BBox3D(bounds=((0, 1), (0, 2), (0, 3))),
            0,
            (2, 1, 1),
            False,
        ],
        [
            BBox3D(bounds=((10, 20), (100, 110), (1000, 1010))),
            0,
            (20, 1, 1),
            False,
        ],
    ],
)
def test_get_slice_exc(
    bbox: BBox3D,
    dim: int,
    resolution: Vec3D,
    allow_slice_rounding: bool,
):
    with pytest.raises(ValueError):
        bbox.get_slice(dim=dim, resolution=resolution, allow_slice_rounding=allow_slice_rounding)


@pytest.mark.parametrize(
    "bbox, pad, resolution, expected",
    [
        [
            BBox3D(bounds=((0, 0), (0, 0), (0, 0))),
            (1, 3, 5),
            (1, 1, 1),
            BBox3D(bounds=((-1, 1), (-3, 3), (-5, 5))),
        ],
        [
            BBox3D(bounds=((0, 0), (0, 0), (0, 0))),
            (1, 3, 5),
            (1, 3, 5),
            BBox3D(bounds=((-1, 1), (-9, 9), (-25, 25))),
        ],
        [
            BBox3D(bounds=((0, 0), (0, 0), (0, 0))),
            ((0, 1), (1, 3), (2, 5)),
            (1, 3, 5),
            BBox3D(bounds=((0, 1), (-3, 9), (-10, 25))),
        ],
    ],
)
def test_pad(bbox: BBox3D, pad, resolution: Vec3D, expected: BBox3D):
    result = bbox.padded(pad=pad, resolution=resolution)
    assert result == expected


@pytest.mark.parametrize(
    "bbox, pad, resolution, expected_exc",
    [
        [
            BBox3D(bounds=((0, 0), (0, 0), (0, 0))),
            (1, 3, 5, 3),
            (1, 1, 1),
            ValueError,
        ],
    ],
)
def test_pad_exc(bbox: BBox3D, pad, resolution: Vec3D, expected_exc):
    with pytest.raises(expected_exc):
        bbox.padded(pad=pad, resolution=resolution)


@pytest.mark.parametrize(
    "bbox, num_splits, expected",
    [
        [
            BBox3D(bounds=((0, 0), (0, 0), (0, 0))),
            (1, 2, 3),
            [
                BBox3D(bounds=((0, 0), (0, 0), (0, 0))),
                BBox3D(bounds=((0, 0), (0, 0), (0, 0))),
                BBox3D(bounds=((0, 0), (0, 0), (0, 0))),
                BBox3D(bounds=((0, 0), (0, 0), (0, 0))),
                BBox3D(bounds=((0, 0), (0, 0), (0, 0))),
                BBox3D(bounds=((0, 0), (0, 0), (0, 0))),
            ],
        ],
        [
            BBox3D(bounds=((0, 3), (0, 4), (0, 5))),
            (3, 1, 5),
            [
                BBox3D(bounds=((0, 1), (0, 4), (0, 1))),
                BBox3D(bounds=((0, 1), (0, 4), (1, 2))),
                BBox3D(bounds=((0, 1), (0, 4), (2, 3))),
                BBox3D(bounds=((0, 1), (0, 4), (3, 4))),
                BBox3D(bounds=((0, 1), (0, 4), (4, 5))),
                BBox3D(bounds=((1, 2), (0, 4), (0, 1))),
                BBox3D(bounds=((1, 2), (0, 4), (1, 2))),
                BBox3D(bounds=((1, 2), (0, 4), (2, 3))),
                BBox3D(bounds=((1, 2), (0, 4), (3, 4))),
                BBox3D(bounds=((1, 2), (0, 4), (4, 5))),
                BBox3D(bounds=((2, 3), (0, 4), (0, 1))),
                BBox3D(bounds=((2, 3), (0, 4), (1, 2))),
                BBox3D(bounds=((2, 3), (0, 4), (2, 3))),
                BBox3D(bounds=((2, 3), (0, 4), (3, 4))),
                BBox3D(bounds=((2, 3), (0, 4), (4, 5))),
            ],
        ],
        [
            BBox3D(bounds=((0, 2), (-3, 0), (0, 0))),
            (2, 2, 1),
            [
                BBox3D(bounds=((0, 1), (-3, -1.5), (0, 0))),
                BBox3D(bounds=((0, 1), (-1.5, 0), (0, 0))),
                BBox3D(bounds=((1, 2), (-3, -1.5), (0, 0))),
                BBox3D(bounds=((1, 2), (-1.5, 0), (0, 0))),
            ],
        ],
    ],
)
def test_split(bbox: BBox3D, num_splits: Sequence[int], expected: BBox3D):
    result = bbox.split(num_splits=num_splits)
    assert result == expected


@pytest.mark.parametrize(
    "bbox, num_splits, expected_exc",
    [
        [
            BBox3D(bounds=((0, 0), (0, 0), (0, 0))),
            (1, 3, 5, 3),
            ValueError,
        ],
    ],
)
def test_num_splits_exc(bbox: BBox3D, num_splits: Sequence[int], expected_exc):
    with pytest.raises(expected_exc):
        bbox.split(num_splits=num_splits)


@pytest.mark.parametrize(
    "bbox, offset, resolution, expected",
    [
        [
            BBox3D(bounds=((0, 1), (0, 1), (0, 1))),
            (10, 10, 10),
            (1, 1, 1),
            BBox3D(bounds=((10, 11), (10, 11), (10, 11))),
        ],
    ],
)
def test_translated(
    bbox: BBox3D,
    offset: Vec3D,
    resolution: Vec3D,
    expected: BBox3D,
):
    result = bbox.translated(offset, resolution)
    assert result == expected


def test_translated_exc():
    bbox = BBox3D(bounds=((10, 20), (100, 110), (1000, 1010)))
    with pytest.raises(ValueError):
        bbox.translated(resolution=(1, 2), offset=(1, 2, 3))


@pytest.mark.parametrize(
    "bbox, offset, resolution, expected",
    [
        [
            BBox3D(bounds=((0, 10), (0, 10), (0, 10))),
            (2, 2, 2),
            (1, 1, 1),
            BBox3D(bounds=((2, 10), (2, 10), (2, 10))),
        ],
    ],
)
def test_translated_start(
    bbox: BBox3D,
    offset: Vec3D,
    resolution: Vec3D,
    expected: BBox3D,
):
    result = bbox.translated_start(offset, resolution)
    assert result == expected


def test_translated_start_exc():
    bbox = BBox3D(bounds=((10, 20), (100, 110), (1000, 1010)))
    with pytest.raises(ValueError):
        bbox.translated_start(resolution=(1, 2), offset=(1, 2, 3))


@pytest.mark.parametrize(
    "bbox, offset, resolution, expected",
    [
        [
            BBox3D(bounds=((0, 10), (0, 10), (0, 10))),
            (2, 2, 2),
            (1, 1, 1),
            BBox3D(bounds=((0, 12), (0, 12), (0, 12))),
        ],
    ],
)
def test_translated_end(
    bbox: BBox3D,
    offset: Vec3D,
    resolution: Vec3D,
    expected: BBox3D,
):
    result = bbox.translated_end(offset, resolution)
    assert result == expected


def test_translated_end_exc():
    bbox = BBox3D(bounds=((10, 20), (100, 110), (1000, 1010)))
    with pytest.raises(ValueError):
        bbox.translated_end(resolution=(1, 2), offset=(1, 2, 3))


@pytest.mark.parametrize(
    "bbox, crop, resolution, expected",
    [
        [
            BBox3D(bounds=((-1, 1), (-3, 3), (-5, 5))),
            (1, 3, 5),
            (1, 1, 1),
            BBox3D(bounds=((0, 0), (0, 0), (0, 0))),
        ],
        [
            BBox3D(bounds=((-1, 1), (-9, 9), (-25, 25))),
            (1, 3, 5),
            (1, 3, 5),
            BBox3D(bounds=((0, 0), (0, 0), (0, 0))),
        ],
        [
            BBox3D(bounds=((0, 1), (-3, 9), (-10, 25))),
            ((0, 1), (1, 3), (2, 5)),
            (1, 3, 5),
            BBox3D(bounds=((0, 0), (0, 0), (0, 0))),
        ],
    ],
)
def test_cropped(bbox: BBox3D, crop, resolution: Vec3D, expected: BBox3D):
    result = bbox.cropped(crop=crop, resolution=resolution)
    assert result == expected


@pytest.mark.parametrize(
    "bbox, crop, resolution, expected_exc",
    [
        [
            BBox3D(bounds=((0, 0), (0, 0), (0, 0))),
            (1, 3, 5, 3),
            (1, 1, 1),
            ValueError,
        ],
    ],
)
def test_cropped_exc(bbox: BBox3D, crop, resolution: Vec3D, expected_exc):
    with pytest.raises(expected_exc):
        bbox.cropped(crop=crop, resolution=resolution)


@pytest.mark.parametrize(
    "bbox, grid_offset, grid_size, mode, expected",
    [
        [
            BBox3D(bounds=((-1, 1), (-9, 9), (-25, 25))),
            (1, 2, 3),
            (1, 3, 5),
            "shrink",
            BBox3D(bounds=((-1, 1), (-7, 8), (-22, 23))),
        ],
        [
            BBox3D(bounds=((-1, 1), (-3, 3), (-5, 5))),
            (0, -1, -2),
            (1, 2, 3),
            "expand",
            BBox3D(bounds=((-1, 1), (-3, 3), (-5, 7))),
        ],
        [
            BBox3D(bounds=((-1, 1), (-3, 3.1), (-5, 7))),
            (0, 0.0, 0.0),
            (0.1, 0.6 / 3.0, 0.1 + 0.2),
            "expand",
            BBox3D(bounds=((-1.0, 1.0), (-3.0, 3.2), (-5.1, 7.2))),
        ],
    ],
)
def test_snapped(bbox: BBox3D, grid_offset, grid_size, mode, expected: BBox3D):
    result = bbox.snapped(grid_offset=grid_offset, grid_size=grid_size, mode=mode)
    assert result.start.allclose(expected.start)
    assert result.end.allclose(expected.end)


@pytest.mark.parametrize(
    "bbox, grid_offset, grid_size, mode",
    [
        [
            BBox3D(bounds=((-1, 1), (-3, 3), (-5, 5))),
            (0, -1, -2),
            (1, 2, 3),
            "badmode",
        ],
        [
            BBox3D(bounds=((-1, 1), (-3, 3), (-5, 5))),
            (1, 2, 3),
            (1, 2, 3, 4),
            "badmode",
        ],
        [
            BBox3D(bounds=((-1, 1), (-3, 3), (-5, 5))),
            (1, 2, 3),
            (1, 2, 3, 4),
            "badmode",
        ],
    ],
)
def test_snapped_exc(bbox: BBox3D, grid_offset, grid_size, mode):
    with pytest.raises(Exception):
        bbox.snapped(grid_offset=grid_offset, grid_size=grid_size, mode=mode)


@pytest.mark.parametrize(
    "bbox1, bbox2, expected",
    [
        [
            BBox3D(bounds=((-1, 1), (-9, 9), (-25, 25))),
            BBox3D(bounds=((0, 2), (3, 5), (7, 42))),
            (False, False, False, False, False, False),
        ],
        [
            BBox3D(bounds=((-1, 1), (-3, 3), (-5, 5))),
            BBox3D(bounds=((1, 2), (-3, 3), (-5, 7))),
            (False, False, True, True, True, False),
        ],
        [
            BBox3D(bounds=((-1, 1), (-9, 9), (-25, 25))),
            BBox3D(bounds=((0, 1), (-10, 9), (-50, 40))),
            (False, True, False, True, False, False),
        ],
    ],
)
def test_aligned(bbox1: BBox3D, bbox2: BBox3D, expected: BBox3D):
    assert bbox1.aligned(bbox2) == expected
    assert bbox2.aligned(bbox1) == expected


@pytest.mark.parametrize(
    "bbox1, bbox2, expected",
    [
        [
            BBox3D(bounds=((-1, 1), (-9, 9), (-25, 25))),
            BBox3D(bounds=((0, 2), (3, 5), (7, 42))),
            False,
        ],
        [
            BBox3D(bounds=((-1, 1), (-3, 3), (-5, 5))),
            BBox3D(bounds=((1, 2), (-3, 3), (-5, 7))),
            False,
        ],
        [
            BBox3D(bounds=((-1, 1), (-9, 9), (-25, 25))),
            BBox3D(bounds=((-2, 1), (-10, 9), (-50, 40))),
            True,
        ],
    ],
)
def test_contained_in(bbox1: BBox3D, bbox2: BBox3D, expected: BBox3D):
    assert bbox1.contained_in(bbox2) == expected


@pytest.mark.parametrize(
    "bbox1, bbox2, expected",
    [
        [
            BBox3D(bounds=((-1, 1), (-9, 9), (-25, 25))),
            BBox3D(bounds=((0, 2), (3, 5), (7, 42))),
            True,
        ],
        [
            BBox3D(bounds=((-1, 1), (-3, 3), (-5, 5))),
            BBox3D(bounds=((1, 2), (-3, 3), (-5, 7))),
            False,
        ],
    ],
)
def test_intersects(bbox1: BBox3D, bbox2: BBox3D, expected: BBox3D):
    assert bbox1.intersects(bbox2) == expected
    assert bbox2.intersects(bbox1) == expected


@pytest.mark.parametrize(
    "bbox1, bbox2, expected",
    [
        [
            BBox3D(bounds=((-1, 1), (-9, 9), (-25, 25))),
            BBox3D(bounds=((0, 2), (3, 5), (7, 42))),
            BBox3D(bounds=((0, 1), (3, 5), (7, 25))),
        ],
        [
            BBox3D(bounds=((-1, 1), (-3, 3), (-5, 5))),
            BBox3D(bounds=((1, 2), (-3, 3), (-5, 7))),
            BBox3D(bounds=((0, 0), (0, 0), (0, 0))),
        ],
    ],
)
def test_intersection(bbox1: BBox3D, bbox2: BBox3D, expected: BBox3D):
    assert bbox1.intersection(bbox2) == expected


@pytest.mark.parametrize(
    "bbox1, bbox2, expected",
    [
        [
            BBox3D(bounds=((-1, 1), (-9, 9), (-25, 25))),
            BBox3D(bounds=((0, 2), (3, 5), (7, 42))),
            BBox3D(bounds=((-1, 2), (-9, 9), (-25, 42))),
        ],
        [
            BBox3D(bounds=((-1, 1), (-3, 3), (-5, 5))),
            BBox3D(bounds=((1, 2), (-3, 3), (-5, 7))),
            BBox3D(bounds=((-1, 2), (-3, 3), (-5, 7))),
        ],
    ],
)
def test_supremum(bbox1: BBox3D, bbox2: BBox3D, expected: BBox3D):
    assert bbox1.supremum(bbox2) == expected


@pytest.mark.parametrize(
    "bbox, point, resolution, expected",
    [
        [BBox3D(bounds=((-1, 1), (-9, 9), (-25, 25))), Vec3D(0, -9, 24), (1, 1, 1), True],
        [BBox3D(bounds=((1, 2), (-3, 3), (-5, 7))), Vec3D(1, 0, -5), (1, 1, 1), True],
        [BBox3D(bounds=((1, 2), (-3, 3), (-5, 7))), Vec3D(0.9, 0, -5), (1, 1, 1), False],
        [BBox3D(bounds=((1, 2), (-3, 3), (-5, 7))), Vec3D(1, 0, 7), (1, 1, 1), False],
        [BBox3D(bounds=((1, 2), (-3, 3), (-5, 7))), Vec3D(0.1, 0, -0.5), (10, 10, 10), True],
        [BBox3D(bounds=((1, 2), (-3, 3), (-5, 7))), Vec3D(0.09, 0, -0.5), (10, 10, 10), False],
    ],
)
def test_contains(bbox: BBox3D, point: Vec3D, resolution: Vec3D, expected: bool):
    assert bbox.contains(point, resolution) == expected


def test_contains_exc():
    bbox = BBox3D(bounds=((1, 2), (-3, 3), (-5, 7)))
    with pytest.raises(ValueError):
        bbox.contains((1, 2, 3, 4), (1, 2, 3))
    with pytest.raises(ValueError):
        bbox.contains((1, 2, 3), (1, 2))


def test_transpose_local():
    bbox = BBox3D(bounds=((0, 1), (3, 5), (7, 25)))
    transposed = bbox.transposed(0, 1, local=True)
    expected = BBox3D(bounds=((0, 2), (3, 4), (7, 25)))
    assert transposed == expected


def test_transpose_global():
    bbox = BBox3D(bounds=((0, 1), (3, 5), (7, 25)))
    transposed = bbox.transposed(0, 1, local=False)
    expected = BBox3D(bounds=((3, 5), (0, 1), (7, 25)))
    assert transposed == expected
