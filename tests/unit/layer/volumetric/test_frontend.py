# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,too-few-public-methods
from __future__ import annotations

import numpy as np
import pytest
import torch

from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex
from zetta_utils.layer.volumetric.conversion import (
    convert_idx,
    convert_write,
    get_bbox_from_user_vol_idx,
    get_desired_res_from_user_vol_idx,
)


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
def test_convert_idx(
    kwargs: dict,
    idx,
    expected: VolumetricIndex,
):
    index_resolution = kwargs.get("index_resolution", None)
    default_desired_resolution = kwargs.get("default_desired_resolution", None)
    allow_slice_rounding = kwargs.get("allow_slice_rounding", False)

    result = convert_idx(idx, index_resolution, default_desired_resolution, allow_slice_rounding)
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
def test_convert_idx_exc(
    kwargs: dict,
    idx,
    expected_exc,
):
    index_resolution = kwargs.get("index_resolution", None)
    default_desired_resolution = kwargs.get("default_desired_resolution", None)
    allow_slice_rounding = kwargs.get("allow_slice_rounding", False)

    with pytest.raises(expected_exc):
        convert_idx(idx, index_resolution, default_desired_resolution, allow_slice_rounding)


@pytest.mark.parametrize(
    "data_input, expected_dtype",
    [
        (5.0, np.dtype("float32")),
        (5, np.dtype("int32")),
        (True, np.dtype("int32")),
        (np.array([1, 2, 3]), np.dtype("int64")),
        (torch.tensor([1, 2, 3]), np.dtype("int64")),
    ],
)
def test_convert_write(data_input, expected_dtype):
    idx = VolumetricIndex(
        bbox=BBox3D.from_slices((slice(0, 1), slice(0, 4), slice(0, 9))),
        resolution=Vec3D(1, 2, 3),
    )

    _, data_result = convert_write(
        idx,
        data_input,
        index_resolution=None,
        default_desired_resolution=None,
        allow_slice_rounding=False,
    )

    assert data_result.dtype == expected_dtype


def test_get_bbox_from_user_vol_idx():
    # Test with VolumetricIndex
    idx = VolumetricIndex(
        bbox=BBox3D.from_slices((slice(0, 1), slice(0, 4), slice(0, 9))),
        resolution=Vec3D(1, 2, 3),
    )
    result = get_bbox_from_user_vol_idx(idx, None)
    assert result == idx.bbox

    # Test with tuple of (resolution, bbox)
    resolution = Vec3D(1, 2, 3)
    bbox = BBox3D.from_slices((slice(0, 1), slice(0, 4), slice(0, 9)))
    result = get_bbox_from_user_vol_idx((resolution, bbox), None)
    assert result == bbox

    # Test with bbox only and index_resolution
    index_resolution = Vec3D(1, 2, 3)
    result = get_bbox_from_user_vol_idx(bbox, index_resolution)
    assert result == bbox


def test_get_desired_res_from_user_vol_idx():
    # Test with VolumetricIndex
    idx = VolumetricIndex(
        bbox=BBox3D.from_slices((slice(0, 1), slice(0, 4), slice(0, 9))),
        resolution=Vec3D(1, 2, 3),
    )
    result = get_desired_res_from_user_vol_idx(idx, None)
    assert result == idx.resolution

    # Test with tuple of (resolution, bbox)
    resolution = Vec3D(1, 2, 3)
    bbox = BBox3D.from_slices((slice(0, 1), slice(0, 4), slice(0, 9)))
    result = get_desired_res_from_user_vol_idx((resolution, bbox), None)
    assert result == resolution

    # Test with bbox only and default_desired_resolution
    default_resolution = Vec3D(4, 5, 6)
    result = get_desired_res_from_user_vol_idx(bbox, default_resolution)
    assert result == default_resolution
