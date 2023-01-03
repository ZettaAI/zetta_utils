# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,unsubscriptable-object
from zetta_utils.training.datasets.sample_indexers import VolumetricNGLIndexer
from zetta_utils.typing import IntVec3D, Vec3D


def test_len(mocker):
    mocker.patch(
        "zetta_utils.parsing.ngl_state.read_remote_annotations", return_value=[Vec3D(0, 0, 0)]
    )
    indexer = VolumetricNGLIndexer(
        "dummy_path", chunk_size=IntVec3D(8, 8, 1), resolution=Vec3D(1, 1, 1)
    )
    assert len(indexer) == 1


def test_call(mocker):
    mocker.patch(
        "zetta_utils.parsing.ngl_state.read_remote_annotations", return_value=[Vec3D(0, 0, 0)]
    )
    indexer = VolumetricNGLIndexer(
        "dummy_path", chunk_size=IntVec3D(8, 8, 1), resolution=Vec3D(1, 1, 1)
    )
    idx = indexer(0)
    assert idx[0] is None
    assert idx[1].start == -3
    assert idx[1].stop == 5
    assert idx[-1].start == 0
    assert idx[-1].stop == 1


def test_desired_res(mocker):
    mocker.patch(
        "zetta_utils.parsing.ngl_state.read_remote_annotations", return_value=[Vec3D(0, 0, 0)]
    )
    desired_res = Vec3D(10, 10, 10)
    indexer = VolumetricNGLIndexer(
        "dummy_path",
        chunk_size=IntVec3D(8, 8, 1),
        resolution=Vec3D(1, 1, 1),
        desired_resolution=desired_res,
        index_resolution=Vec3D(0.5, 0.5, 0.5),
    )
    idx = indexer(0)
    assert idx[0] == desired_res
    assert idx[1].start == -6
    assert idx[1].stop == 10
    assert idx[-1].start == 0
    assert idx[-1].stop == 2
