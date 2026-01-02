# pylint: disable=missing-docstring,invalid-name
import numpy as np
import pytest
import torch

from zetta_utils.tensor_ops import filtering

from ..helpers import assert_array_equal


@pytest.mark.parametrize(
    "data,kwargs",
    [
        [np.array([[[[1.0, 1.0, 4.0, 1.0, 1.0]]]], dtype=np.float32), {"sigma": 2.0}],
        [torch.tensor([[[[1.0, 2.0, 100.0, 3.0, 4.0]]]], dtype=torch.float32), {"threshold": 5.0}],
    ],
)
def test_nanmedian_filter_type_and_mode(data, kwargs):
    result = filtering.nanmedian_filter(data, **kwargs)
    assert type(result) is type(data)
    assert np.isnan(result[0, 0, 0, 2])
    assert result[0, 0, 0, 0] == data[0, 0, 0, 0]


def test_nanmedian_filter_fill_value():
    data = np.array([[[[1.0, 1.0, 100.0, 1.0, 1.0]]]], dtype=np.float32)
    result = filtering.nanmedian_filter(data, threshold=5.0, fill_value=42.0)
    assert result[0, 0, 0, 2] == 42.0


def test_nanmedian_filter_no_outliers():
    data = np.array([[[[1.0, 2.0, 3.0, 4.0, 5.0]]]], dtype=np.float32)
    result = filtering.nanmedian_filter(data, threshold=5.0)
    assert_array_equal(result, data)


def test_nanmedian_filter_with_existing_nans():
    data = np.array([[[[1.0, np.nan, 1.0, 1.0, 100.0]]]], dtype=np.float32)
    result = filtering.nanmedian_filter(data, threshold=5.0)
    assert np.isnan(result[0, 0, 0, 1])  # original NaN
    assert np.isnan(result[0, 0, 0, 4])  # outlier replaced


def test_nanmedian_filter_multichannel_voxel_masking():
    data = np.array(
        [[[[1.0, 1.0, 100.0, 1.0]]], [[[1.0, 1.0, 1.0, 1.0]]]],
        dtype=np.float32,
    )
    result = filtering.nanmedian_filter(data, threshold=5.0)
    # both channels at z=2 masked (voxel-wise)
    assert np.isnan(result[0, 0, 0, 2])
    assert np.isnan(result[1, 0, 0, 2])


def test_nanmedian_filter_axis():
    data = np.ones((1, 5, 1, 1), dtype=np.float32)
    data[0, 2, 0, 0] = 100.0
    result = filtering.nanmedian_filter(data, axis=1, threshold=5.0)
    assert np.isnan(result[0, 2, 0, 0])


def test_nanmedian_filter_exc():
    data = np.ones((1, 2, 2, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="at least one of"):
        filtering.nanmedian_filter(data)
