# pylint: disable=missing-docstring
import pytest
import numpy as np
import torch

from zetta_utils.data import basic_ops
from .utils import assert_array_equal


@pytest.mark.parametrize(
    "data, dim, expected_shape",
    [
        [np.ones((1, 2)), 0, (1, 1, 2)],
        [np.ones((1, 2)), 2, (1, 2, 1)],
        [np.ones((1, 2)), (0, 3), (1, 1, 2, 1)],
        [torch.ones((1, 2)).int(), 0, (1, 1, 2)],
        [torch.ones((1, 2)).int(), 2, (1, 2, 1)],
    ],
)
def test_unsqueeze(data, dim, expected_shape):
    result = basic_ops.unsqueeze(data, dim)
    assert result.shape == expected_shape


@pytest.mark.parametrize(
    "data, dim",
    [
        [torch.ones((1, 2)), (0, 1)],
    ],
)
def test_unsqueeze_exc(data, dim):
    with pytest.raises(ValueError):
        basic_ops.unsqueeze(data, dim)


@pytest.mark.parametrize(
    "data, dim, expected_shape",
    [
        [np.ones((1, 2, 1)), 0, (2, 1)],
        [np.ones((1, 2, 1)), 2, (1, 2)],
        [np.ones((1, 2, 1)), (0, 2), (2,)],
        [torch.ones((1, 2, 1)), 0, (2, 1)],
        [torch.ones((1, 2, 1)), 2, (1, 2)],
    ],
)
def test_squeeze(data, dim, expected_shape):
    result = basic_ops.squeeze(data, dim)
    assert result.shape == expected_shape


@pytest.mark.parametrize(
    "data, dim",
    [
        [torch.ones((1, 2, 1)), (0, 2)],
    ],
)
def test_squeeze_exc(data, dim):
    with pytest.raises(ValueError):
        basic_ops.squeeze(data, dim)


@pytest.fixture
def array_1d_x0():
    return np.array(
        [
            [
                [1, 0, 1, 1],
            ]
        ],
        dtype=np.float32,
    )


@pytest.fixture
def array_1d_x0_avg_pool():
    return np.array(
        [
            [
                [0.5, 1],
            ]
        ],
        dtype=np.float32,
    )


@pytest.fixture
def array_2d_x0():
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
def array_2d_x0_avg_pool():
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


@pytest.fixture
def array_x1():
    return np.array(
        [
            [
                [
                    [
                        [1, 0, 1, 1],
                        [0, 0, 1, 2],
                        [0, 2, 0, 0],
                        [0, 2, 0, 4],
                    ]
                ]
            ]
        ],
        dtype=np.float32,
    )


@pytest.fixture
def array_x1_avg_pool():
    return np.array(
        [
            [
                [
                    [
                        [0.25, 1.25],
                        [1, 1],
                    ]
                ]
            ]
        ],
        dtype=np.float32,
    )


@pytest.mark.parametrize(
    "data_name, mode, kwargs, expected_name",
    [
        ["array_1d_x0", "img", {"scale_factor": 0.5}, "array_1d_x0_avg_pool"],
        ["array_2d_x0", "img", {"scale_factor": 0.5}, "array_2d_x0_avg_pool"],
        ["array_2d_x0", "img", {"scale_factor": [0.5, 0.5]}, "array_2d_x0_avg_pool"],
        [
            "array_2d_x0",
            "img",
            {"scale_factor": None, "size": [2, 2]},
            "array_2d_x0_avg_pool",
        ],
        ["array_2d_x0", "bilinear", {"scale_factor": 0.5}, "array_2d_x0_avg_pool"],
        ["array_x1", "img", {"scale_factor": 0.5}, "array_x1_avg_pool"],
        ["mask_x0", "mask", {"scale_factor": 0.5}, "mask_x0_max_pool"],
        [
            "mask_x0",
            "mask",
            {"scale_factor": 0.5, "mask_value_thr": 0.5},
            "mask_x0_max_pool_thr05",
        ],
        ["mask_x0", "mask", {"scale_factor": 2.0}, "mask_x0_ups"],
        ["field_x0", "field", {"scale_factor": 2.0}, "field_x0_ups"],
        ["field_x0", "field", {"scale_factor": [2.0, 2.0]}, "field_x0_ups"],
        ["seg_x0", "segmentation", {"scale_factor": 2.0}, "seg_x0_ups"],
        ["torch_seg_x0", "segmentation", {"scale_factor": 2.0}, "torch_seg_x0_ups"],
    ],
)
def test_interpolate(data_name, mode, kwargs, expected_name, request):
    data = request.getfixturevalue(data_name)
    result = basic_ops.interpolate(data, mode=mode, **kwargs)
    expected = request.getfixturevalue(expected_name)
    assert_array_equal(result, expected)


@pytest.fixture
def array_6d():
    return np.array(
        [
            [
                [
                    [
                        [
                            [1, 0, 1, 1],
                            [0, 0, 1, 2],
                            [0, 2, 0, 0],
                            [0, 2, 0, 4],
                        ]
                    ]
                ]
            ]
        ],
        dtype=np.float32,
    )


@pytest.mark.parametrize(
    "data_name, mode, kwargs, expected_exc",
    [
        ["array_2d_x0", "img", {}, ValueError],
        ["array_6d", "img", {"scale_factor": 0.5}, RuntimeError],
        ["array_1d_x0", "img", {"scale_factor": 0.357}, RuntimeError],
    ],
)
def test_interpolate_exc(data_name, mode, kwargs, request, expected_exc):
    data = request.getfixturevalue(data_name)
    with pytest.raises(expected_exc):
        basic_ops.interpolate(data, mode=mode, **kwargs)


@pytest.mark.parametrize(
    "data, mode, operand, kwargs, expected",
    [
        [
            np.array(([0, 1], [2, 3])),
            "==",
            0,
            {},
            np.array(([[True, False], [False, False]])),
        ],
        [
            np.array(([0, 1], [2, 3])),
            "!=",
            0,
            {},
            np.array(([[False, True], [True, True]])),
        ],
        [
            np.array(([0, 1], [2, 3])),
            ">",
            2,
            {},
            np.array(([[False, False], [False, True]])),
        ],
        [
            np.array(([0, 1], [2, 3])),
            ">=",
            2,
            {},
            np.array(([[False, False], [True, True]])),
        ],
        [
            np.array(([0, 1], [2, 3])),
            "<",
            1,
            {},
            np.array(([[True, False], [False, False]])),
        ],
        [
            np.array(([0, 1], [2, 3])),
            "<=",
            1,
            {},
            np.array(([[True, True], [False, False]])),
        ],
        [
            np.array(([0, 1], [2, 3])),
            "<=",
            1,
            {"binarize": False, "fill": 5},
            np.array(([[5, 5], [2, 3]])),
        ],
    ],
)
def test_compare(data, mode, operand, kwargs, expected):
    result = basic_ops.compare(data, mode, operand, **kwargs)
    assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "data, mode, operand, kwargs",
    [
        [
            np.array(([0, 1], [2, 3])),
            "<=",
            1,
            {"binarize": False, "fill": None},
        ],
        [
            np.array(([0, 1], [2, 3])),
            "<=",
            1,
            {"binarize": True, "fill": 1},
        ],
    ],
)
def test_compare_exc(data, mode, operand, kwargs):
    with pytest.raises(ValueError):
        basic_ops.compare(data, mode, operand, **kwargs)
