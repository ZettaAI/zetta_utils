from typing import Callable, Optional, Union

import numpy as np
import pytest
import torch

from zetta_utils import augmentations, distributions
from zetta_utils.tensor_typing import Tensor, TensorTypeVar

from ..helpers import assert_array_equal


@pytest.mark.parametrize(
    "data, value_distr, mask_fn, expected",
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
    value_distr: Union[distributions.Distribution, float],
    mask_fn: Optional[Callable[..., Tensor]],
    expected: TensorTypeVar,
):
    result = augmentations.tensor.add_scalar_aug(
        data=data, value_distr=value_distr, mask_fn=mask_fn
    )
    assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "data, low_distr, high_distr, mask_fn, expected",
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
    low_distr: Optional[Union[distributions.Distribution, float]],
    high_distr: Optional[Union[distributions.Distribution, float]],
    mask_fn: Optional[Callable[..., Tensor]],
    expected: TensorTypeVar,
):
    result = augmentations.tensor.clamp_values_aug(
        data=data, low_distr=low_distr, high_distr=high_distr, mask_fn=mask_fn
    )
    assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "data, tile_size, tile_stride, max_brightness_change, "
    "rotation_degree, preserve_data_val, repeats, expected",
    [
        [
            np.array(
                [
                    [
                        [[0], [0], [0], [0]],
                        [[0], [0], [0], [0]],
                        [[0], [0], [0], [0]],
                        [[0], [0], [0], [0]],
                    ]
                ]
            ),
            1,
            2,
            1,
            0,
            None,
            1,
            np.array(
                [
                    [
                        [[1], [0], [1], [0]],
                        [[0], [0], [0], [0]],
                        [[1], [0], [1], [0]],
                        [[0], [0], [0], [0]],
                    ]
                ]
            ),
        ],
        [
            np.array(
                [
                    [
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ]
                ]
            ),
            1,
            2,
            1,
            0,
            None,
            1,
            np.array(
                [
                    [
                        [1, 0, 1, 0],
                        [0, 0, 0, 0],
                        [1, 0, 1, 0],
                        [0, 0, 0, 0],
                    ]
                ]
            ),
        ],
        [
            np.array(
                [
                    [
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ]
                ]
            ),
            1,
            1,
            1,
            0,
            None,
            1,
            np.array(
                [
                    [
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                    ]
                ]
            ),
        ],
        [
            np.array(
                [
                    [
                        [-99, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ]
                ]
            ),
            1,
            1,
            1,
            0,
            -99,
            1,
            np.array(
                [
                    [
                        [-99, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                    ]
                ]
            ),
        ],
        [
            np.array(
                [
                    [
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ]
                ]
            ),
            1,
            2,
            1,
            45,
            None,
            1,
            np.array(
                [
                    [
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [1, 0, 0, 1],
                        [0, 1, 1, 0],
                    ]
                ]
            ),
        ],
    ],
)
def test_square_tile_pattern_aug(
    data: TensorTypeVar,
    tile_size: Union[distributions.Distribution, float],
    tile_stride: Union[distributions.Distribution, float],
    max_brightness_change: Union[distributions.Distribution, float],
    rotation_degree: Union[distributions.Distribution, float],
    preserve_data_val: Optional[float],
    repeats: int,
    expected: TensorTypeVar,
    mocker,
):
    mocker.patch("random.choice", return_value=1.0)
    mocker.patch("random.randint", return_value=0)
    mocker.patch("zetta_utils.distributions.uniform_distr", return_value=lambda: 1.0)

    result = augmentations.tensor.square_tile_pattern_aug(
        data=data,
        tile_size=tile_size,
        tile_stride=tile_stride,
        max_brightness_change=max_brightness_change,
        rotation_degree=rotation_degree,
        preserve_data_val=preserve_data_val,
        repeats=repeats,
    )
    assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "shape, device, expected_shape, expected_device",
    [
        [(1, 16, 16, 3), "cpu", torch.Size([1, 16, 16, 3]), torch.device("cpu")],
        [[2, 16, 16, 2], "cpu", torch.Size([2, 16, 16, 2]), torch.device("cpu")],
        [(3, 16, 16, 1), None, torch.Size([3, 16, 16, 1]), torch.device("cpu")],
    ],
)
def test_rand_perlin_2d(shape, device, expected_shape, expected_device):
    result = augmentations.tensor.rand_perlin_2d(shape, res=[4, 4], device=device)
    assert result.shape == expected_shape
    assert result.device == expected_device


@pytest.mark.parametrize(
    "shape, device, expected_shape, expected_device",
    [
        [(1, 16, 16, 3), "cpu", torch.Size([1, 16, 16, 3]), torch.device("cpu")],
        [[2, 16, 16, 2], "cpu", torch.Size([2, 16, 16, 2]), torch.device("cpu")],
        [(3, 16, 16, 1), None, torch.Size([3, 16, 16, 1]), torch.device("cpu")],
    ],
)
def test_rand_perlin_2d_octaves(shape, device, expected_shape, expected_device):
    result = augmentations.tensor.rand_perlin_2d_octaves(
        shape, res=[4, 4], octaves=2, device=device
    )
    assert result.shape == expected_shape
    assert result.device == expected_device


@pytest.mark.parametrize(
    "shape, res",
    [
        [(1, 16, 16), [8, 8]],
        [[2, 16, 16, 2], (4,)],
    ],
)
def test_rand_perlin_2d_failure(shape, res):
    with pytest.raises(ValueError):
        augmentations.tensor.rand_perlin_2d(shape, res)

@pytest.mark.parametrize(
    "shape, res",
    [
        [(1, 16, 16), [8, 8]],
        [[2, 16, 16, 2], (4,)],
    ],
)
def test_rand_perlin_2d_octaves_failure(shape, res):
    with pytest.raises(ValueError):
        augmentations.tensor.rand_perlin_2d_octaves(shape, res)
