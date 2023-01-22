# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,too-few-public-methods
import pytest

from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric import VolumetricFrontend, VolumetricIndex


@pytest.mark.parametrize(
    "kwargs, idx, expected",
    [
        [
            {},
            VolumetricIndex(
                bbox=BBox3D.from_slices((slice(0, 1), slice(0, 4), slice(0, 9))),
                resolution=Vec3D(1, 2, 3),
            ),
            VolumetricIndex(
                bbox=BBox3D.from_slices((slice(0, 1), slice(0, 4), slice(0, 9))),
                resolution=Vec3D(1, 2, 3),
            ),
        ],
        [
            {},
            (Vec3D(1, 2, 3), BBox3D.from_slices((slice(0, 1), slice(0, 4), slice(0, 9)))),
            VolumetricIndex(
                bbox=BBox3D.from_slices((slice(0, 1), slice(0, 4), slice(0, 9))),
                resolution=Vec3D(1, 2, 3),
            ),
        ],
        [
            {"default_desired_resolution": Vec3D(1, 2, 3)},
            BBox3D.from_slices((slice(0, 1), slice(0, 4), slice(0, 9))),
            VolumetricIndex(
                bbox=BBox3D.from_slices((slice(0, 1), slice(0, 4), slice(0, 9))),
                resolution=Vec3D(1, 2, 3),
            ),
        ],
        [
            {},
            (Vec3D(1, 2, 3), slice(0, 1), slice(0, 2), slice(0, 3)),
            VolumetricIndex(
                bbox=BBox3D.from_slices((slice(0, 1), slice(0, 4), slice(0, 9))),
                resolution=Vec3D(1, 2, 3),
            ),
        ],
        [
            {},
            (Vec3D(1, 2, 3), (slice(0, 1), slice(0, 2), slice(0, 3))),
            VolumetricIndex(
                bbox=BBox3D.from_slices((slice(0, 1), slice(0, 4), slice(0, 9))),
                resolution=Vec3D(1, 2, 3),
            ),
        ],
        [
            {"default_desired_resolution": Vec3D(8, 8, 8), "index_resolution": Vec3D(1, 2, 3)},
            (slice(0, 1), slice(0, 2), slice(0, 3)),
            VolumetricIndex(
                bbox=BBox3D.from_slices((slice(0, 1), slice(0, 4), slice(0, 9))),
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
            {"index_resolution": Vec3D(1, 2, 3)},
            (None, slice(0, 2), slice(0, 2), slice(0, 2)),
            ValueError,
        ],
        [
            {},
            (None, BBox3D.from_slices((slice(0, 1), slice(0, 4), slice(0, 9)))),
            ValueError,
        ],
        [
            {"default_desired_resolution": Vec3D(1, 2, 3)},
            (slice(0, 1), slice(0, 2), slice(0, 3)),
            ValueError,
        ],
        [
            {"index_resolution": Vec3D(2, 2, 3)},
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
        convert._convert_idx(idx)
