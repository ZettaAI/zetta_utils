from typing import Union, Optional, Callable
import pytest

import numpy as np

from zetta_utils.typing import Tensor, Number, TensorTypeVar
from zetta_utils import augmentations, distributions

from ..helpers import assert_array_equal


@pytest.mark.parametrize(
    "data, value, mask_fn, expected",
    [
        [
            np.array([0, 1, 2]),
            1,
            None,
            np.array([1, 2, 3]),
        ],
        [
            np.array([0, 1, 2]),
            1,
            lambda x: x == 2,
            np.array([0, 1, 3]),
        ],
        [
            np.array([0, 1, 2]),
            1,
            lambda x: x,
            np.array([0, 2, 4]),
        ],
    ],
)
def test_brightness(
    data: TensorTypeVar,
    value: Union[distributions.Distribution, Number],
    mask_fn: Optional[Callable[..., Tensor]],
    expected: TensorTypeVar,
):
    result = augmentations.tensor.add_scalar_aug(data=data, value=value, mask_fn=mask_fn)
    assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "data, low, high, mask_fn, expected",
    [
        [
            np.array([0, 1, 2]),
            None,
            None,
            None,
            np.array([0, 1, 2]),
        ],
        [
            np.array([0, 1, 2]),
            1,
            None,
            lambda x: x < 2,
            np.array([1, 1, 2]),
        ],
        [
            np.array([0, 1, 2]),
            1,
            1,
            lambda x: x != 0,
            np.array([0, 1, 1]),
        ],
    ],
)
def test_clamp_values(
    data: TensorTypeVar,
    low: Optional[Union[distributions.Distribution, Number]],
    high: Optional[Union[distributions.Distribution, Number]],
    mask_fn: Optional[Callable[..., Tensor]],
    expected: TensorTypeVar,
):
    result = augmentations.tensor.clamp_values_aug(data=data, low=low, high=high, mask_fn=mask_fn)
    assert_array_equal(result, expected)
