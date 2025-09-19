import numpy as np
import pytest

from zetta_utils.tensor_ops import projection

from ..helpers import assert_array_equal


@pytest.fixture
def projection_data():
    return np.array(
        [
            [
                [[0, 1, 2], [0, 0, 3]],
                [[0, 4, 0], [5, 0, 0]],
            ]
        ],
        dtype=np.float32,
    )


@pytest.fixture
def projection_data_with_nan():
    return np.array(
        [
            [
                [[np.nan, 1, 2], [np.nan, np.nan, 3]],
                [[np.nan, 4, np.nan], [5, np.nan, np.nan]],
            ]
        ],
        dtype=np.float32,
    )


@pytest.fixture
def projection_expected_first():
    return np.array(
        [
            [
                [[1], [3]],
                [[4], [5]],
            ]
        ],
        dtype=np.float32,
    )


@pytest.fixture
def projection_expected_last():
    return np.array(
        [
            [
                [[2], [3]],
                [[4], [5]],
            ]
        ],
        dtype=np.float32,
    )


@pytest.fixture
def projection_data_all_bg():
    return np.zeros((1, 2, 2, 3), dtype=np.float32)


@pytest.fixture
def projection_expected_all_bg():
    return np.zeros((1, 2, 2, 1), dtype=np.float32)


@pytest.mark.parametrize(
    "data_name, kwargs, expected_name",
    [
        [
            "projection_data",
            {"bg_color": 0.0, "axis": 3, "direction": "first"},
            "projection_expected_first",
        ],
        [
            "projection_data_with_nan",
            {"bg_color": np.nan, "axis": 3, "direction": "first"},
            "projection_expected_first",
        ],
        [
            "projection_data",
            {"bg_color": 0.0, "axis": 3, "direction": "last"},
            "projection_expected_last",
        ],
        [
            "projection_data_all_bg",
            {"bg_color": 0.0, "axis": 3, "direction": "first"},
            "projection_expected_all_bg",
        ],
    ],
)
def test_first_hit_projection(data_name, kwargs, expected_name, request):
    data = request.getfixturevalue(data_name)
    result = projection.first_hit_projection(data, **kwargs)
    expected = request.getfixturevalue(expected_name)
    assert_array_equal(result, expected)
