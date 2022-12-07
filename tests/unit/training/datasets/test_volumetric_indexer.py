# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,unsubscriptable-object
from zetta_utils.bcube import BoundingCube
from zetta_utils.training.datasets.sample_indexers import VolumetricStridedIndexer
from zetta_utils.typing import Vec3D


def test_len(mocker):
    vsi = VolumetricStridedIndexer(
        bcube=BoundingCube.from_coords(
            start_coord=Vec3D((0, 0, 0)), end_coord=Vec3D((1, 2, 3)), resolution=Vec3D((1, 1, 1))
        ),
        chunk_size=Vec3D((1, 1, 1)),
        resolution=Vec3D((1, 1, 1)),
        stride=Vec3D((1, 1, 1)),
    )
    assert len(vsi) == 6


def test_call_default(mocker):
    vsi = VolumetricStridedIndexer(
        bcube=BoundingCube.from_coords(
            start_coord=Vec3D((0, 0, 0)), end_coord=Vec3D((1, 2, 3)), resolution=Vec3D((1, 1, 1))
        ),
        chunk_size=Vec3D((1, 1, 1)),
        resolution=Vec3D((1, 1, 1)),
        stride=Vec3D((1, 1, 1)),
    )
    assert vsi(0) == (None, slice(0, 1), slice(0, 1), slice(0, 1))


def test_call_index_res(mocker):
    vsi = VolumetricStridedIndexer(
        bcube=BoundingCube.from_coords(
            start_coord=Vec3D((0, 0, 0)), end_coord=Vec3D((1, 2, 3)), resolution=Vec3D((1, 1, 1))
        ),
        chunk_size=Vec3D((1, 1, 1)),
        resolution=Vec3D((1, 1, 1)),
        stride=Vec3D((1, 1, 1)),
        index_resolution=Vec3D((0.5, 0.5, 0.5)),
    )
    assert vsi(0) == (None, slice(0, 2), slice(0, 2), slice(0, 2))


def test_call_desired_res(mocker):
    vsi = VolumetricStridedIndexer(
        bcube=BoundingCube.from_coords(
            start_coord=Vec3D((0, 0, 0)), end_coord=Vec3D((1, 2, 3)), resolution=Vec3D((1, 1, 1))
        ),
        chunk_size=Vec3D((1, 1, 1)),
        resolution=Vec3D((1, 1, 1)),
        stride=Vec3D((1, 1, 1)),
        desired_resolution=Vec3D((0.5, 0.5, 0.5)),
    )
    assert vsi(0) == ((0.5, 0.5, 0.5), slice(0, 1), slice(0, 1), slice(0, 1))
