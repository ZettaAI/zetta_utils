# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,unsubscriptable-object,unused-variable
import multiprocessing

import pytest

from zetta_utils.geometry import BBox3D, BBoxStrider, IntVec3D, Vec3D


# basic functionality tests
def test_bbox_rounding(mocker):
    strider = BBoxStrider(
        bbox=BBox3D.from_coords(
            start_coord=Vec3D(0, 0, 0), end_coord=Vec3D(1, 1, 4), resolution=Vec3D(1, 1, 1)
        ),
        chunk_size=IntVec3D(1, 1, 3),
        stride=IntVec3D(1, 1, 3),
        resolution=Vec3D(1, 1, 1),
        mode="shrink",
    )
    assert strider.num_chunks == 1
    assert strider.step_limits == IntVec3D(1, 1, 1)


def test_bbox_strider_get_nth(mocker):
    strider = BBoxStrider(
        bbox=BBox3D.from_coords(
            start_coord=Vec3D(0, 0, 0), end_coord=Vec3D(1, 2, 3), resolution=Vec3D(1, 1, 1)
        ),
        chunk_size=IntVec3D(1, 1, 1),
        resolution=Vec3D(1, 1, 1),
        stride=IntVec3D(1, 1, 1),
    )
    assert strider.get_nth_chunk_bbox(0) == BBox3D.from_slices(
        (slice(0, 1), slice(0, 1), slice(0, 1))
    )
    assert strider.get_nth_chunk_bbox(1) == BBox3D.from_slices(
        (slice(0, 1), slice(1, 2), slice(0, 1))
    )
    assert strider.get_nth_chunk_bbox(4) == BBox3D.from_slices(
        (slice(0, 1), slice(0, 1), slice(2, 3))
    )


def test_bbox_strider_get_all_chunks(mocker):
    strider = BBoxStrider(
        bbox=BBox3D.from_coords(
            start_coord=Vec3D(0, 0, 0), end_coord=Vec3D(1, 1, 2), resolution=Vec3D(1, 1, 1)
        ),
        chunk_size=IntVec3D(1, 1, 1),
        stride=IntVec3D(1, 1, 1),
        resolution=Vec3D(1, 1, 1),
    )
    assert strider.get_all_chunk_bboxes() == [
        BBox3D.from_slices((slice(0, 1), slice(0, 1), slice(0, 1))),
        BBox3D.from_slices((slice(0, 1), slice(0, 1), slice(1, 2))),
    ]


def test_bbox_strider_get_all_chunks_parallel(mocker):
    num_cores = multiprocessing.cpu_count()
    strider = BBoxStrider(
        bbox=BBox3D.from_coords(
            start_coord=Vec3D(0, 0, 0), end_coord=Vec3D(2, 1, num_cores), resolution=Vec3D(1, 1, 1)
        ),
        chunk_size=IntVec3D(1, 1, 1),
        stride=IntVec3D(1, 1, 1),
        resolution=Vec3D(1, 1, 1),
    )
    assert strider.get_all_chunk_bboxes()[0:2] == [
        BBox3D.from_slices((slice(0, 1), slice(0, 1), slice(0, 1))),
        BBox3D.from_slices((slice(1, 2), slice(0, 1), slice(0, 1))),
    ]


@pytest.mark.parametrize(
    "start_coord, end_coord, resolution, chunk_size, stride, stride_start_offset_in_unit, mode, max_superchunk_size, expected",
    [
        [
            Vec3D(0, 0, 0),
            Vec3D(1, 2, 3),
            Vec3D(1, 1, 1),
            IntVec3D(1, 1, 1),
            IntVec3D(1, 1, 1),
            None,
            "shrink",
            None,
            6,
        ],
        [
            Vec3D(0, 0, 0),
            Vec3D(1, 2, 3),
            Vec3D(1, 1, 1),
            IntVec3D(2, 2, 2),
            IntVec3D(2, 2, 2),
            IntVec3D(0, 0, 0),
            "expand",
            None,
            2,
        ],
        [
            Vec3D(0, -1, 0),
            Vec3D(1, 2, 3),
            Vec3D(1, 1, 1),
            IntVec3D(2, 2, 2),
            IntVec3D(2, 2, 2),
            IntVec3D(0, 0, 0),
            "exact",
            None,
            4,
        ],
        [
            Vec3D(0, -1, 0),
            Vec3D(1, 2, 3),
            Vec3D(1, 1, 1),
            IntVec3D(2, 2, 2),
            IntVec3D(2, 2, 2),
            None,
            "exact",
            None,
            4,
        ],
        [
            Vec3D(0, -1, 0),
            Vec3D(4, 4, 5),
            Vec3D(1, 1, 1),
            IntVec3D(2, 2, 2),
            IntVec3D(2, 2, 2),
            IntVec3D(-2, -2, 0),
            "exact",
            IntVec3D(4, 4, 4),
            4,
        ],
        [
            Vec3D(0, 0, 0),
            Vec3D(4, 5, 6),
            Vec3D(1, 1, 1),
            IntVec3D(3, 3, 3),
            IntVec3D(2, 2, 2),
            None,
            "expand",
            None,
            12,
        ],
        [
            Vec3D(0, 0, 0),
            Vec3D(4, 5, 6),
            Vec3D(1, 1, 1),
            IntVec3D(3, 3, 3),
            IntVec3D(2, 2, 2),
            IntVec3D(-1, 1, -3),
            "expand",
            None,
            18,
        ],
        [
            Vec3D(0, 0, 0),
            Vec3D(4, 5, 6),
            Vec3D(1, 1, 1),
            IntVec3D(2, 2, 2),
            IntVec3D(2, 2, 2),
            None,
            "expand",
            IntVec3D(5, 5, 5),
            4,
        ],
        [
            Vec3D(-4, 0, 0),
            Vec3D(4, 5, 6),
            Vec3D(1, 1, 1),
            IntVec3D(3, 3, 3),
            IntVec3D(2, 2, 2),
            IntVec3D(1, 1, 1),
            "shrink",
            None,
            6,
        ],
    ],
)
def test_bbox_strider_len(
    start_coord,
    end_coord,
    resolution,
    chunk_size,
    stride,
    stride_start_offset_in_unit,
    mode,
    max_superchunk_size,
    expected,
):
    strider = BBoxStrider(
        bbox=BBox3D.from_coords(
            start_coord=start_coord, end_coord=end_coord, resolution=resolution
        ),
        chunk_size=chunk_size,
        resolution=resolution,
        stride=stride,
        stride_start_offset_in_unit=stride_start_offset_in_unit,
        mode=mode,
        max_superchunk_size=max_superchunk_size,
    )
    assert strider.num_chunks == expected


@pytest.mark.parametrize(
    "start_coord, end_coord, resolution, chunk_size, stride, stride_start_offset_in_unit, mode, max_superchunk_size, idx, expected",
    [
        [
            Vec3D(0, 0, 0),
            Vec3D(1, 2, 3),
            Vec3D(1, 1, 1),
            IntVec3D(1, 1, 1),
            IntVec3D(1, 1, 1),
            None,
            "shrink",
            None,
            1,
            BBox3D.from_slices((slice(0, 1), slice(1, 2), slice(0, 1))),
        ],
        [
            Vec3D(0, 0, 0),
            Vec3D(1, 2, 3),
            Vec3D(1, 1, 1),
            IntVec3D(2, 2, 2),
            IntVec3D(2, 2, 2),
            IntVec3D(0, 0, 0),
            "expand",
            None,
            1,
            BBox3D.from_slices((slice(0, 2), slice(0, 2), slice(2, 4))),
        ],
        [
            Vec3D(0, -1, 0),
            Vec3D(1, 2, 3),
            Vec3D(1, 1, 1),
            IntVec3D(2, 2, 2),
            IntVec3D(2, 2, 2),
            IntVec3D(0, 0, 0),
            "exact",
            None,
            0,
            BBox3D.from_slices((slice(0, 1), slice(-1, 0), slice(0, 2))),
        ],
        [
            Vec3D(0, -1, 0),
            Vec3D(1, 2, 3),
            Vec3D(1, 1, 1),
            IntVec3D(2, 2, 2),
            IntVec3D(2, 2, 2),
            None,
            "exact",
            None,
            0,
            BBox3D.from_slices((slice(0, 1), slice(-1, 1), slice(0, 2))),
        ],
        [
            Vec3D(0, -1, 0),
            Vec3D(4, 4, 5),
            Vec3D(1, 1, 1),
            IntVec3D(2, 2, 2),
            IntVec3D(2, 2, 2),
            IntVec3D(-2, -2, 0),
            "exact",
            IntVec3D(4, 4, 4),
            0,
            BBox3D.from_slices((slice(0, 4), slice(-1, 0), slice(0, 4))),
        ],
        [
            Vec3D(0, 0, 0),
            Vec3D(4, 5, 6),
            Vec3D(1, 1, 1),
            IntVec3D(3, 3, 3),
            IntVec3D(2, 2, 2),
            None,
            "expand",
            None,
            11,
            BBox3D.from_slices((slice(2, 5), slice(2, 5), slice(4, 7))),
        ],
        [
            Vec3D(0, 0, 0),
            Vec3D(4, 5, 6),
            Vec3D(1, 1, 1),
            IntVec3D(3, 3, 3),
            IntVec3D(2, 2, 2),
            IntVec3D(-1, 1, -3),
            "expand",
            None,
            0,
            BBox3D.from_slices((slice(-1, 2), slice(-1, 2), slice(-1, 2))),
        ],
        [
            Vec3D(0, 0, 0),
            Vec3D(4, 5, 6),
            Vec3D(1, 1, 1),
            IntVec3D(2, 2, 2),
            IntVec3D(2, 2, 2),
            None,
            "expand",
            IntVec3D(5, 5, 5),
            3,
            BBox3D.from_slices((slice(0, 4), slice(4, 6), slice(4, 6))),
        ],
        [
            Vec3D(-4, 0, 0),
            Vec3D(4, 5, 6),
            Vec3D(1, 1, 1),
            IntVec3D(3, 3, 3),
            IntVec3D(2, 2, 2),
            IntVec3D(1, 1, 1),
            "shrink",
            None,
            5,
            BBox3D.from_slices((slice(1, 4), slice(1, 4), slice(3, 6))),
        ],
    ],
)
def test_bbox_strider_get_nth_res(
    start_coord,
    end_coord,
    resolution,
    chunk_size,
    stride,
    stride_start_offset_in_unit,
    mode,
    max_superchunk_size,
    idx,
    expected,
):
    strider = BBoxStrider(
        bbox=BBox3D.from_coords(
            start_coord=start_coord, end_coord=end_coord, resolution=resolution
        ),
        chunk_size=chunk_size,
        resolution=resolution,
        stride=stride,
        stride_start_offset_in_unit=stride_start_offset_in_unit,
        mode=mode,
        max_superchunk_size=max_superchunk_size,
    )
    assert strider.get_nth_chunk_bbox(idx) == expected


def test_bbox_strider_exc(mocker):
    bbox = BBox3D.from_coords(
        start_coord=Vec3D(0, 0, 0), end_coord=Vec3D(1, 1, 2), resolution=Vec3D(1, 1, 1)
    )

    # check exact doesn't work with stride != chunk_size
    with pytest.raises(NotImplementedError):
        strider = BBoxStrider(
            bbox=bbox,
            resolution=Vec3D(1, 1, 1),
            chunk_size=IntVec3D(1, 1, 1),
            stride=IntVec3D(2, 2, 2),
            mode="exact",
        )

    # check superchunking doesn't work with stride != chunk_size
    with pytest.raises(NotImplementedError):
        strider = BBoxStrider(
            bbox=bbox,
            resolution=Vec3D(1, 1, 1),
            chunk_size=IntVec3D(1, 1, 1),
            stride=IntVec3D(2, 2, 2),
            max_superchunk_size=IntVec3D(3, 3, 3),
        )

    # check superchunking doesn't work when max_superchunk_size is not at least as large as the chunk_size
    with pytest.raises(ValueError):
        strider = BBoxStrider(
            bbox=bbox,
            resolution=Vec3D(1, 1, 1),
            chunk_size=IntVec3D(2, 2, 2),
            stride=IntVec3D(2, 2, 2),
            max_superchunk_size=IntVec3D(1, 1, 1),
        )
