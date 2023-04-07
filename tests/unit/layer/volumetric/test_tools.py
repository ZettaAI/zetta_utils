# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,too-few-public-methods
import pytest

from zetta_utils.geometry import BBox3D, IntVec3D, Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex, VolumetricIndexChunker


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
    res = vic(idx=idx, stride_start_offset_in_unit=stride_start_offset_in_unit, mode=mode)
    assert res[1].bbox == chunk1
