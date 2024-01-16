# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,unsubscriptable-object
import pytest

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


@pytest.mark.parametrize(
    "coord_nm, chunk_res, chunk_size, expected_bounds",
    [
        [
            Vec3D(0, 0, 0),
            Vec3D(1, 1, 1),
            Vec3D[int](8, 8, 1),
            BBox3D(bounds=((-4, 4), (-4, 4), (0, 1))),
        ],
        [
            Vec3D(158515.2, 510771.2, 1440),
            Vec3D(34.4, 34.4, 45),
            Vec3D[int](1024, 1024, 1),
            BBox3D(
                bounds=(
                    (4096 * 34.4, 5120 * 34.4),
                    (14336 * 34.4, 15360 * 34.4),
                    (32 * 45, 33 * 45),
                )
            ),
        ],
    ],
)
def test_call(coord_nm, chunk_res, chunk_size, expected_bounds, mocker):
    mocker.patch("zetta_utils.parsing.ngl_state.read_remote_annotations", return_value=[coord_nm])
    indexer = VolumetricNGLIndexer("dummy_path", chunk_size=chunk_size, resolution=chunk_res)
    idx = indexer(0)
    assert idx.resolution == chunk_res
    assert idx.bbox == expected_bounds
