# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,too-few-public-methods
from typing import Literal, Tuple, Iterable, Callable
import pytest

from zetta_utils.typing import Vec3D
from zetta_utils.data.processors import Interpolate
from zetta_utils.data.layers.indexers.volumetric import VolumetricIndexer, RawVolumetricIndex, VolumetricIndex


@pytest.mark.parametrize(
    "indexer, idx, expected",
    [
        [
            VolumetricIndexer(),
            ((1, 2, 3), slice(0, 1), slice(0, 2), slice(0, 3)),
            (
                VolumetricIndex(
                    slices=(slice(0, 1), slice(0, 2), slice(0, 3)),
                    resolution=(1, 2, 3)
                ),
                (1, 2, 3)
            )
        ],
        [
            VolumetricIndexer(index_resolution=(1, 2, 3)),
            (slice(0, 1), slice(0, 2), slice(0, 3)),
            (
                VolumetricIndex(
                    slices=(slice(0, 1), slice(0, 2), slice(0, 3)),
                    resolution=(1, 2, 3)
                ),
                (1, 2, 3)
            )
        ],
        [
            VolumetricIndexer(data_resolution=(5, 5, 5)),
            ((1, 1, 1), slice(0, 10), slice(0, 10), slice(0, 10)),
            (
                VolumetricIndex(
                    slices=(slice(0, 2), slice(0, 2), slice(0, 2)),
                    resolution=(5, 5, 5)
                ),
                (1, 1, 1)
            )
        ],
        [
            VolumetricIndexer(data_resolution=(5, 5, 5), index_resolution=(1, 1, 1)),
            (slice(0, 10), slice(0, 10), slice(0, 10)),
            (
                VolumetricIndex(
                    slices=(slice(0, 2), slice(0, 2), slice(0, 2)),
                    resolution=(5, 5, 5)
                ),
                (5, 5, 5)
            )
        ]
    ],
)
def test_process_idx(
    indexer: VolumetricIndexer,
    idx: RawVolumetricIndex,
    expected: Tuple[VolumetricIndex, Vec3D]
):
    result = indexer._process_idx(idx)
    assert result == expected

@pytest.mark.parametrize(
    "indexer, idx, expected_exc",
    [
        [
            VolumetricIndexer(data_resolution=(2, 2, 2)),
            (None, slice(0, 2), slice(0, 2), slice(0, 2)),
            ValueError
        ],

    ],
)
def test_process_idx_exc(
    indexer: VolumetricIndexer,
    idx: RawVolumetricIndex,
    expected_exc,
):
    with pytest.raises(expected_exc):
        indexer._process_idx(idx)


@pytest.mark.parametrize(
    "indexer, idx, mode, expected",
    [
        [
            VolumetricIndexer(),
            ((1, 2, 3), slice(0, 1), slice(0, 2), slice(0, 3)),
            'read',
            (
                VolumetricIndex(
                    slices=(slice(0, 1), slice(0, 2), slice(0, 3)),
                    resolution=(1, 2, 3)
                ), []
            )
        ],
        [
            VolumetricIndexer(data_resolution=(2, 2, 2), interpolation_mode='img'),
            ((1, 1, 1), slice(0, 2), slice(0, 2), slice(0, 2)),
            'read',
            (
                VolumetricIndex(
                    slices=(slice(0, 1), slice(0, 1), slice(0, 1)),
                    resolution=(2, 2, 2)
                ),
                [Interpolate(interpolation_mode='img', scale_factor=(2.0, 2.0, 2.0))]
            )
        ],
        [
            VolumetricIndexer(data_resolution=(2, 2, 2), interpolation_mode='img'),
            ((1, 1, 1), slice(0, 2), slice(0, 2), slice(0, 2)),
            'write',
            (
                VolumetricIndex(
                    slices=(slice(0, 1), slice(0, 1), slice(0, 1)),
                    resolution=(2, 2, 2)
                ),
                [Interpolate(interpolation_mode='img', scale_factor=(0.5, 0.5, 0.5))]
            )
        ],

    ],
)
def test_call(
    indexer: VolumetricIndexer,
    idx: RawVolumetricIndex,
    mode: Literal['read', 'write'],
    expected: Tuple[VolumetricIndex, Iterable[Callable]]
):
    result = indexer(idx, mode)
    assert result == expected



@pytest.mark.parametrize(
    "indexer, idx, mode, expected_exc",
    [
        [
            VolumetricIndexer(data_resolution=(2, 2, 2), interpolation_mode=None),
            ((1, 1, 1), slice(0, 2), slice(0, 2), slice(0, 2)),
            'write',
            RuntimeError
        ],

    ],
)
def test_call_exc(
    indexer: VolumetricIndexer,
    idx: RawVolumetricIndex,
    mode: Literal['read', 'write'],
    expected_exc,
):
    with pytest.raises(expected_exc):
        indexer(idx, mode)
