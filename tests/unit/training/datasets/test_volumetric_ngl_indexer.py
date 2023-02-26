# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,unsubscriptable-object
from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.training.datasets.sample_indexers import VolumetricNGLIndexer


def test_len(mocker):
    mocker.patch(
        "zetta_utils.parsing.ngl_state.read_remote_annotations", return_value=[Vec3D(0, 0, 0)]
    )
    indexer = VolumetricNGLIndexer(
        "dummy_path", chunk_size=Vec3D[int](8, 8, 1), resolution=Vec3D(1, 1, 1)
    )
    assert len(indexer) == 1


def test_call(mocker):
    mocker.patch(
        "zetta_utils.parsing.ngl_state.read_remote_annotations", return_value=[Vec3D(0, 0, 0)]
    )
    indexer = VolumetricNGLIndexer(
        "dummy_path", chunk_size=Vec3D[int](8, 8, 1), resolution=Vec3D(1, 1, 1)
    )
    idx = indexer(0)
    assert idx.resolution == Vec3D(1, 1, 1)
    assert idx.bbox == BBox3D(bounds=((-3, 5), (-3, 5), (0, 1)))
