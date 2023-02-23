# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,unsubscriptable-object
from zetta_utils.geometry import BBox3D, IntVec3D, Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex
from zetta_utils.training.datasets.sample_indexers import VolumetricStridedIndexer


def test_len(mocker):
    vsi = VolumetricStridedIndexer(
        bbox=BBox3D.from_coords(
            start_coord=Vec3D(0, 0, 0), end_coord=Vec3D(1, 2, 3), resolution=Vec3D(1, 1, 1)
        ),
        chunk_size=IntVec3D(1, 1, 1),
        resolution=Vec3D(1, 1, 1),
        stride=IntVec3D(1, 1, 1),
    )
    assert len(vsi) == 6


def test_call_default(mocker):
    vsi = VolumetricStridedIndexer(
        bbox=BBox3D.from_coords(
            start_coord=Vec3D(0, 0, 0), end_coord=Vec3D(1, 2, 3), resolution=Vec3D(1, 1, 1)
        ),
        chunk_size=IntVec3D(1, 1, 1),
        resolution=Vec3D(1, 1, 1),
        stride=IntVec3D(1, 1, 1),
    )
    assert vsi(0) == VolumetricIndex(Vec3D(1, 1, 1), BBox3D(bounds=((0, 1), (0, 1), (0, 1))))
