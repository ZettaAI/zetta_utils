# pylint: disable=missing-docstring
import numpy as np
import pytest
import torch

from zetta_utils.tensor_ops import convert


@pytest.mark.parametrize("x, expected", [[np.ones(3), np.ones(3)], [torch.ones(3), np.ones(3)]])
def test_to_np(x, expected):
    result = convert.to_np(x)
    np.testing.assert_array_equal(result, expected)


def test_to_np_exc():
    with pytest.raises(Exception):
        convert.to_np("hello")  # type: ignore


@pytest.mark.parametrize(
    "x, expected",
    [[torch.ones(3), torch.ones(3)], [np.ones(3, dtype=np.float32), torch.ones(3)]],
)
def test_to_torch(x, expected):
    result = convert.to_torch(x)
    assert torch.equal(result, expected)


@pytest.mark.parametrize(
    "x, expected",
    [
        [torch.ones(3).byte(), torch.ones(3).float()],
        [np.ones(3).astype(np.uint8), np.ones(3).astype(np.float32)],
    ],
)
def test_to_float(x, expected):
    result = convert.to_float32(x)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "x, expected",
    [
        [torch.ones(3).float(), torch.ones(3).byte()],
        [np.ones(3).astype(np.float32), np.ones(3).astype(np.uint8)],
    ],
)
def test_to_uint(x, expected):
    result = convert.to_uint8(x)
    np.testing.assert_array_equal(result, expected)
