# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,too-few-public-methods
import pytest

from zetta_utils.bcube import BoundingCube
from zetta_utils.layer.volumetric import (
    RawVolumetricIndex,
    VolumetricIndex,
    VolumetricIndexConverter,
)


@pytest.mark.parametrize(
    "kwargs, idx, expected",
    [
        [
            {},
            VolumetricIndex(
                bcube=BoundingCube.from_slices((slice(0, 1), slice(0, 4), slice(0, 9))),
                resolution=(1, 2, 3),
            ),
            VolumetricIndex(
                bcube=BoundingCube.from_slices((slice(0, 1), slice(0, 4), slice(0, 9))),
                resolution=(1, 2, 3),
            ),
        ],
        [
            {},
            ((1, 2, 3), BoundingCube.from_slices((slice(0, 1), slice(0, 4), slice(0, 9)))),
            VolumetricIndex(
                bcube=BoundingCube.from_slices((slice(0, 1), slice(0, 4), slice(0, 9))),
                resolution=(1, 2, 3),
            ),
        ],
        [
            {"default_desired_resolution": (1, 2, 3)},
            BoundingCube.from_slices((slice(0, 1), slice(0, 4), slice(0, 9))),
            VolumetricIndex(
                bcube=BoundingCube.from_slices((slice(0, 1), slice(0, 4), slice(0, 9))),
                resolution=(1, 2, 3),
            ),
        ],
        [
            {},
            ((1, 2, 3), slice(0, 1), slice(0, 2), slice(0, 3)),
            VolumetricIndex(
                bcube=BoundingCube.from_slices((slice(0, 1), slice(0, 4), slice(0, 9))),
                resolution=(1, 2, 3),
            ),
        ],
        [
            {},
            ((1, 2, 3), (slice(0, 1), slice(0, 2), slice(0, 3))),
            VolumetricIndex(
                bcube=BoundingCube.from_slices((slice(0, 1), slice(0, 4), slice(0, 9))),
                resolution=(1, 2, 3),
            ),
        ],
        [
            {"default_desired_resolution": (8, 8, 8), "index_resolution": (1, 2, 3)},
            (slice(0, 1), slice(0, 2), slice(0, 3)),
            VolumetricIndex(
                bcube=BoundingCube.from_slices((slice(0, 1), slice(0, 4), slice(0, 9))),
                resolution=(8, 8, 8),
            ),
        ],
    ],
)
def test_volumetric_conver(
    kwargs: dict,
    idx: RawVolumetricIndex,
    expected: VolumetricIndex,
):
    conver = VolumetricIndexConverter(**kwargs)
    result = conver(idx)
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
    idx: RawVolumetricIndex,
    expected_exc,
):
    conver = VolumetricIndexConverter(**kwargs)
    with pytest.raises(expected_exc):
        conver(idx)
