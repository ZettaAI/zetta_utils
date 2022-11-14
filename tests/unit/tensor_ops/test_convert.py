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
    with pytest.raises(TypeError):
        convert.to_np("hello")


@pytest.mark.parametrize(
    "x, expected",
    [[torch.ones(3), torch.ones(3)], [np.ones(3, dtype=np.float32), torch.ones(3)]],
)
def test_to_torch(x, expected):
    result = convert.to_torch(x)
    assert torch.equal(result, expected)
