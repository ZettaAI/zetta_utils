# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,unsubscriptable-object
import pytest

from zetta_utils.geometry import BBox3D, IntVec3D, Vec3D
from zetta_utils.training.datasets.sample_indexers import (
    ChainIndexer,
    VolumetricStridedIndexer,
)


def _gen_chain():
    vsi_1 = VolumetricStridedIndexer(
        bbox=BBox3D.from_coords(
            start_coord=Vec3D(0, 0, 0), end_coord=Vec3D(1, 2, 3), resolution=Vec3D(1, 1, 1)
        ),
        chunk_size=IntVec3D(1, 1, 1),
        resolution=Vec3D(1, 1, 1),
        stride=IntVec3D(1, 1, 1),
    )
    vsi_2 = VolumetricStridedIndexer(
        bbox=BBox3D.from_coords(
            start_coord=Vec3D(10, 10, 10), end_coord=Vec3D(10, 10, 10), resolution=Vec3D(1, 1, 1)
        ),
        chunk_size=IntVec3D(1, 1, 1),
        resolution=Vec3D(1, 1, 1),
        stride=IntVec3D(1, 1, 1),
    )
    vsi_3 = VolumetricStridedIndexer(
        bbox=BBox3D.from_coords(
            start_coord=Vec3D(10, 10, 10), end_coord=Vec3D(11, 12, 12), resolution=Vec3D(1, 1, 1)
        ),
        chunk_size=IntVec3D(1, 1, 1),
        resolution=Vec3D(1, 1, 1),
        stride=IntVec3D(1, 1, 1),
    )
    return ChainIndexer([vsi_1, vsi_2, vsi_3])


def test_len(mocker):
    vsi_chain = _gen_chain()
    assert len(vsi_chain) == 10


def test_call(mocker):
    vsi_chain = _gen_chain()
    assert vsi_chain(0) == (None, slice(0, 1), slice(0, 1), slice(0, 1))
    assert vsi_chain(6) == (None, slice(10, 11), slice(10, 11), slice(10, 11))


def test_call_error(mocker):
    vsi_chain = _gen_chain()
    with pytest.raises(ValueError):
        vsi_chain(10)
    with pytest.raises(ValueError):
        vsi_chain(-1)
