# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,too-few-public-methods
from typing import Literal, Tuple, Iterable, Callable
import pytest

from zetta_utils.partial import ComparablePartial
from zetta_utils.io.indexes.volumetric import (
    VolumetricIndexConverter,
    RawVolumetricIndex,
    VolumetricIndex,
    AdjustDataResolution,
    TranslateVolumetricIndex,
)
from zetta_utils import tensor_ops


@pytest.mark.parametrize(
    "indexer, idx, expected",
    [
        [
            VolumetricIndexConverter(),
            ((1, 2, 3), slice(0, 1), slice(0, 2), slice(0, 3)),
            VolumetricIndex(slices=(slice(0, 1), slice(0, 2), slice(0, 3)), resolution=(1, 2, 3)),
        ],
        [
            VolumetricIndexConverter(
                index_resolution=(1, 2, 3), default_desired_resolution=(1, 1, 1)
            ),
            (slice(0, 1), slice(0, 2), slice(0, 3)),
            VolumetricIndex(slices=(slice(0, 1), slice(0, 4), slice(0, 9)), resolution=(1, 1, 1)),
        ],
        [
            VolumetricIndexConverter(
                default_desired_resolution=(1, 2, 3),
            ),
            (None, slice(0, 1), slice(0, 2), slice(0, 3)),
            VolumetricIndex(slices=(slice(0, 1), slice(0, 2), slice(0, 3)), resolution=(1, 2, 3)),
        ],
    ],
)
def test_volumetric_indexer(
    indexer: VolumetricIndexConverter, idx: RawVolumetricIndex, expected: VolumetricIndex
):
    result = indexer(idx)
    assert result == expected


@pytest.mark.parametrize(
    "indexer, idx, expected_exc",
    [
        [
            VolumetricIndexConverter(
                index_resolution=(1, 2, 3),
            ),
            (None, slice(0, 2), slice(0, 2), slice(0, 2)),
            ValueError,
        ],
    ],
)
def test_volumetric_indexer_exc(
    indexer: VolumetricIndexConverter,
    idx: RawVolumetricIndex,
    expected_exc,
):
    with pytest.raises(expected_exc):
        indexer(idx)


@pytest.mark.parametrize(
    "adj, idx, mode, expected",
    [
        [
            AdjustDataResolution(data_resolution=(1, 2, 3), interpolation_mode="img"),
            VolumetricIndex(slices=(slice(0, 1), slice(0, 2), slice(0, 3)), resolution=(1, 2, 3)),
            "read",
            (
                VolumetricIndex(
                    slices=(slice(0, 1), slice(0, 2), slice(0, 3)), resolution=(1, 2, 3)
                ),
                [],
            ),
        ],
        [
            AdjustDataResolution(data_resolution=(1, 1, 1), interpolation_mode="img"),
            VolumetricIndex(slices=(slice(0, 1), slice(0, 1), slice(0, 1)), resolution=(2, 2, 2)),
            "write",
            (
                VolumetricIndex(
                    slices=(slice(0, 2), slice(0, 2), slice(0, 2)), resolution=(1, 1, 1)
                ),
                [
                    ComparablePartial(
                        tensor_ops.interpolate,
                        mode="img",
                        scale_factor=(2.0, 2.0, 2.0),
                        allow_shape_rounding=False,
                        unsqueeze_to=5,
                    )
                ],
            ),
        ],
        [
            AdjustDataResolution(data_resolution=(1, 1, 1), interpolation_mode="img"),
            VolumetricIndex(slices=(slice(0, 1), slice(0, 1), slice(0, 1)), resolution=(2, 2, 2)),
            "read",
            (
                VolumetricIndex(
                    slices=(slice(0, 2), slice(0, 2), slice(0, 2)), resolution=(1, 1, 1)
                ),
                [
                    ComparablePartial(
                        tensor_ops.interpolate,
                        mode="img",
                        scale_factor=(0.5, 0.5, 0.5),
                        allow_shape_rounding=False,
                        unsqueeze_to=5,
                    )
                ],
            ),
        ],
    ],
)
def test_adjust_data_res(
    adj: AdjustDataResolution,
    idx: VolumetricIndex,
    mode: Literal["read", "write"],
    expected: Tuple[VolumetricIndex, Iterable[Callable]],
):
    result = adj(idx, mode)
    assert result == expected


@pytest.mark.parametrize(
    "adj, idx, expected",
    [
        [
            TranslateVolumetricIndex(offset=(1, 2, 3), resolution=(1, 2, 3)),
            VolumetricIndex(slices=(slice(0, 1), slice(0, 1), slice(0, 1)), resolution=(1, 1, 1)),
            VolumetricIndex(slices=(slice(1, 2), slice(4, 5), slice(9, 10)), resolution=(1, 1, 1)),
        ],
    ],
)
def test_translate_vol_idx(
    adj: TranslateVolumetricIndex,
    idx: VolumetricIndex,
    expected: Tuple[VolumetricIndex, Iterable[Callable]],
):
    result = adj(idx)
    assert result == expected
