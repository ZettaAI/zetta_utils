# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,too-few-public-methods
import pytest

from zetta_utils.geometry import BBox3D, IntVec3D, Vec3D
from zetta_utils.layer.volumetric import (
    VolumetricIndex,
    VolumetricIndexChunker,
    VolumetricIndexStartOffsetOverrider,
)


@pytest.mark.parametrize(
    "resolution, stride, offset, idx, stride_start_offset_in_unit, mode, chunk1",
    [
        [
            Vec3D(1, 1, 1),
            IntVec3D(2, 2, 2),
            IntVec3D(0, 0, 0),
            VolumetricIndex(Vec3D(1, 1, 1), BBox3D(((0, 10), (0, 10), (0, 10)))),
            None,
            "expand",
            BBox3D(((2, 4), (0, 3), (0, 5))),
        ],
        [
            Vec3D(1, 1, 1),
            None,
            IntVec3D(0, 0, 0),
            VolumetricIndex(Vec3D(1, 1, 1), BBox3D(((0, 10), (0, 10), (0, 10)))),
            None,
            "expand",
            BBox3D(((2, 4), (0, 3), (0, 5))),
        ],
        [
            Vec3D(2, 2, 2),
            IntVec3D(2, 2, 2),
            IntVec3D(0, 0, 0),
            VolumetricIndex(Vec3D(1, 1, 1), BBox3D(((0, 10), (0, 10), (0, 10)))),
            None,
            "expand",
            BBox3D(((4, 8), (0, 6), (0, 10))),
        ],
        [
            Vec3D(3, 2, 2),
            IntVec3D(2, 2, 2),
            IntVec3D(2, 2, 2),
            VolumetricIndex(Vec3D(1, 1, 1), BBox3D(((0, 10), (0, 15), (0, 15)))),
            None,
            "expand",
            BBox3D(((6, 12), (8, 14), (4, 14))),
        ],
        [
            Vec3D(3, 2, 2),
            IntVec3D(2, 2, 2),
            IntVec3D(2, 2, 2),
            VolumetricIndex(Vec3D(1, 1, 1), BBox3D(((0, 10), (0, 15), (0, 15)))),
            IntVec3D(1, 2, 3),
            "expand",
            BBox3D(((7, 13), (2, 8), (3, 13))),
        ],
        [
            None,
            IntVec3D(2, 2, 2),
            IntVec3D(0, 0, 0),
            VolumetricIndex(Vec3D(3, 2, 1), BBox3D(((0, 16), (0, 16), (0, 16)))),
            None,
            "expand",
            BBox3D(((6, 12), (0, 6), (0, 5))),
        ],
    ],
)
def test_volumetric_index_chunker(
    resolution, stride, offset, idx, stride_start_offset_in_unit, mode, chunk1
):
    vic = VolumetricIndexChunker(
        chunk_size=IntVec3D(2, 3, 5),
        max_superchunk_size=None,
        stride=stride,
        resolution=resolution,
        offset=offset,
    )
    res = list(vic(idx=idx, stride_start_offset_in_unit=stride_start_offset_in_unit, mode=mode))
    assert res[1].bbox == chunk1


@pytest.mark.parametrize(
    "override_offset, expected_start, expected_stop",
    [
        [[7, 8, 9], [7, 8, 9], [17, 18, 19]],
        [[None, None, 10], [4, 5, 10], [14, 15, 20]],
        [[None, None, None], [4, 5, 6], [14, 15, 16]],
    ],
)
def test_volumetric_index_offset_overrider(override_offset, expected_start, expected_stop):
    index = VolumetricIndex(resolution=Vec3D(1, 1, 1), bbox=BBox3D(((4, 14), (5, 15), (6, 16))))
    visoo = VolumetricIndexStartOffsetOverrider(override_offset=override_offset)
    index = visoo(index)
    assert index.start == Vec3D(*expected_start)
    assert index.stop == Vec3D(*expected_stop)
