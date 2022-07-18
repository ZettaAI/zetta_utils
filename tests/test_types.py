# pylint: disable=missing-docstring
import pytest
import numpy as np
import torch

import zetta_utils as zu


@pytest.mark.parametrize(
    "x, expected", [[np.ones(3), np.ones(3)], [torch.ones(3), np.ones(3)]]
)
def test_to_np(x, expected):
    res = zu.zt_types.to_np(x)
    np.testing.assert_array_equal(res, expected)


def test_to_np_exc():
    with pytest.raises(ValueError):
        zu.zt_types.to_np("hello")
