# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,too-few-public-methods
import pytest

from zetta_utils.bcube import BoundingCube
from zetta_utils.layer.volumetric import VolumetricFrontend, VolumetricIndex
from zetta_utils.typing import Vec3D


@pytest.mark.parametrize(
    "kwargs, idx, expected",
    [
        [
            {},
            VolumetricIndex(
                bcube=BoundingCube.from_slices((slice(0, 1), slice(0, 4), slice(0, 9))),
                resolution=Vec3D(1, 2, 3),
            ),
            VolumetricIndex(
                bcube=BoundingCube.from_slices((slice(0, 1), slice(0, 4), slice(0, 9))),
                resolution=Vec3D(1, 2, 3),
            ),
        ],
        [
            {},
            ((1, 2, 3), BoundingCube.from_slices((slice(0, 1), slice(0, 4), slice(0, 9)))),
            VolumetricIndex(
                bcube=BoundingCube.from_slices((slice(0, 1), slice(0, 4), slice(0, 9))),
                resolution=Vec3D(1, 2, 3),
            ),
        ],
        [
            {"default_desired_resolution": (1, 2, 3)},
            BoundingCube.from_slices((slice(0, 1), slice(0, 4), slice(0, 9))),
            VolumetricIndex(
                bcube=BoundingCube.from_slices((slice(0, 1), slice(0, 4), slice(0, 9))),
                resolution=Vec3D(1, 2, 3),
            ),
        ],
        [
            {},
            ((1, 2, 3), slice(0, 1), slice(0, 2), slice(0, 3)),
            VolumetricIndex(
                bcube=BoundingCube.from_slices((slice(0, 1), slice(0, 4), slice(0, 9))),
                resolution=Vec3D(1, 2, 3),
            ),
        ],
        [
            {},
            ((1, 2, 3), (slice(0, 1), slice(0, 2), slice(0, 3))),
            VolumetricIndex(
                bcube=BoundingCube.from_slices((slice(0, 1), slice(0, 4), slice(0, 9))),
                resolution=Vec3D(1, 2, 3),
            ),
        ],
        [
            {"default_desired_resolution": (8, 8, 8), "index_resolution": (1, 2, 3)},
            (slice(0, 1), slice(0, 2), slice(0, 3)),
            VolumetricIndex(
                bcube=BoundingCube.from_slices((slice(0, 1), slice(0, 4), slice(0, 9))),
                resolution=Vec3D(8, 8, 8),
            ),
        ],
    ],
)
def test_volumetric_convert(
    kwargs: dict,
    idx,
    expected: VolumetricIndex,
):
    convert = VolumetricFrontend(**kwargs)
    result = convert._convert_idx(idx)
    assert result == expected


@pytest.mark.parametrize(
    "kwargs, idx, expected_exc",
    [
        [
            {"index_resolution": (1, 2, 3)},
            (None, slice(0, 2), slice(0, 2), slice(0, 2)),
            ValueError,
        ],
        [
            {},
            (None, BoundingCube.from_slices((slice(0, 1), slice(0, 4), slice(0, 9)))),
            ValueError,
        ],
        [
            {"default_desired_resolution": (1, 2, 3)},
            (slice(0, 1), slice(0, 2), slice(0, 3)),
            ValueError,
        ],
        [
            {"index_resolution": (2, 2, 3)},
            (slice(0, 1), slice(0, 2), slice(0, 3)),
            ValueError,
        ],
    ],
)
def test_volumetric_indexer_exc(
    kwargs: dict,
    idx,
    expected_exc,
):
    convert = VolumetricFrontend(**kwargs)
    with pytest.raises(expected_exc):
        conver._convert_idx(idx)
