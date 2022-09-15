from typing import Union, Optional, Callable
import pytest

import numpy as np

from zetta_utils.typing import Tensor, Number, TensorTypeVar
from zetta_utils import augmentations, distributions

from ..helpers import assert_array_equal


@pytest.mark.parametrize(
    "data, adj, low_cap, high_cap, mask_fn, expected",
    [
        [
            np.array([0, 1, 2]),
            1,
            None,
            None,
            None,
            np.array([1, 2, 3]),
        ],
        [
            np.array([0, 1, 2]),
            1,
            None,
            None,
            lambda x: x == 2,
            np.array([0, 1, 3]),
        ],
        [
            np.array([0, 1, 2]),
            1,
            None,
            1,
            lambda x: x < 2,
            np.array([1, 1, 2]),
        ],
        [
            np.array([0, 1, 2]),
            -0.5,
            0,
            1,
            None,
            np.array([0, 0.5, 1]),
        ],
    ],
)
def test_brightness(
    data: TensorTypeVar,
    adj: Union[distributions.Distribution, Number],
    low_cap: Optional[Union[distributions.Distribution, Number]],
    high_cap: Optional[Union[distributions.Distribution, Number]],
    mask_fn: Optional[Callable[..., Tensor]],
    expected: TensorTypeVar,
):
    result = augmentations.img.brightness_aug(
        data=data, adj=adj, low_cap=low_cap, high_cap=high_cap, mask_fn=mask_fn
    )
    assert_array_equal(result, expected)
