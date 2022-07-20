# pylint: disable=missing-docstring
import pytest
import numpy as np
import torch
import zetta_utils as zu
from .utils import assert_array_equal


@pytest.mark.parametrize(
    "data, dim, expected",
    [
        [np.ones((1, 2)), 0, np.array([[[1, 1]]])],
        [np.ones((1, 2)), 2, np.array([[[1], [1]]])],
        [torch.ones((1, 2)).int(), 0, torch.tensor([[[1, 1]]]).int()],
        [torch.ones((1, 2)).int(), 2, torch.tensor([[[1], [1]]]).int()],
    ],
)
def test_unsqueeze(data, dim, expected):
    result = zu.data.basic_ops.unsqueeze(data, dim)
    assert_array_equal(result, expected)


@pytest.fixture
def array_x0():
    return np.array(
        [
            [
                [
                    [1, 0, 1, 1],
                    [0, 0, 1, 2],
                    [0, 2, 0, 0],
                    [0, 2, 0, 4],
                ]
            ]
        ],
        dtype=np.float32,
    )


@pytest.fixture
def array_x0_avg_pool():
    return np.array(
        [
            [
                [
                    [0.25, 1.25],
                    [1, 1],
                ]
            ]
        ],
        dtype=np.float32,
    )


@pytest.fixture
def mask_x0():
    return np.array(
        [
            [
                [
                    [0, 1, 1, 1],
                    [0, 0, 0, 0],
                    [0, 1, 0, 0],
                    [1, 1, 0, 0],
                ]
            ]
        ],
        dtype=bool,
    )


@pytest.fixture
def mask_x0_max_pool():
    return np.array(
        [
            [
                [
                    [1, 1],
                    [1, 0],
                ]
            ]
        ],
        dtype=bool,
    )


@pytest.fixture
def mask_x0_max_pool_thr05():
    return np.array(
        [
            [
                [
                    [0, 0],
                    [1, 0],
                ]
            ]
        ],
        dtype=bool,
    )


@pytest.fixture
def mask_x0_ups():
    return np.array(
        [
            [
                [
                    [0, 0, 1, 1, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0],
                ]
            ]
        ],
        dtype=bool,
    )


@pytest.fixture
def field_x0():
    return np.array(
        [
            [
                [
                    [0, 0],
                    [1.5, 0],
                ]
            ]
        ],
        dtype=float,
    )


@pytest.fixture
def field_x0_ups():
    return np.array(
        [
            [
                [
                    [0, 0, 0, 0],
                    [0.75, 0.5625, 0.1875, 0],
                    [2.25, 1.6875, 0.5625, 0],
                    [3, 2.25, 0.75, 0],
                ]
            ]
        ],
        dtype=float,
    )


@pytest.fixture
def seg_x0():
    return np.array(
        [
            [
                [
                    [0, 1],
                    [2, 0],
                ]
            ]
        ],
        dtype=int,
    )


@pytest.fixture
def seg_x0_ups():
    return np.array(
        [
            [
                [
                    [0, 0, 1, 1],
                    [0, 0, 1, 1],
                    [2, 2, 0, 0],
                    [2, 2, 0, 0],
                ]
            ]
        ],
        dtype=int,
    )


@pytest.fixture
def torch_seg_x0():
    return torch.tensor(
        [
            [
                [
                    [0, 1],
                    [2, 0],
                ]
            ]
        ]
    ).int()


@pytest.fixture
def torch_seg_x0_ups():
    return torch.tensor(
        [
            [
                [
                    [0, 0, 1, 1],
                    [0, 0, 1, 1],
                    [2, 2, 0, 0],
                    [2, 2, 0, 0],
                ]
            ]
        ]
    ).int()


@pytest.mark.parametrize(
    "data_name, mode, kwargs, expected_name",
    [
        ["array_x0", "img", {"scale_factor": 0.5}, "array_x0_avg_pool"],
        ["array_x0", "bilinear", {"scale_factor": 0.5}, "array_x0_avg_pool"],
        ["mask_x0", "mask", {"scale_factor": 0.5}, "mask_x0_max_pool"],
        [
            "mask_x0",
            "mask",
            {"scale_factor": 0.5, "mask_value_thr": 0.5},
            "mask_x0_max_pool_thr05",
        ],  # pylint: disable=line-too-long
        ["mask_x0", "mask", {"scale_factor": 2.0}, "mask_x0_ups"],
        ["field_x0", "field", {"scale_factor": 2.0}, "field_x0_ups"],
        ["seg_x0", "segmentation", {"scale_factor": 2.0}, "seg_x0_ups"],
        ["torch_seg_x0", "segmentation", {"scale_factor": 2.0}, "torch_seg_x0_ups"],
    ],
)
def test_interpolate(data_name, mode, kwargs, expected_name, request):
    data = request.getfixturevalue(data_name)
    result = zu.data.basic_ops.interpolate(data, mode=mode, **kwargs)
    expected = request.getfixturevalue(expected_name)
    assert_array_equal(result, expected)
