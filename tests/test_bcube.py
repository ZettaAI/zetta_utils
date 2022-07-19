# pylint: disable=missing-docstring
import pytest

from zetta_utils.bcube import BoundingCube


@pytest.mark.parametrize(
    "kwargs, expected_x, expected_y, expected_z",
    [
        [{"slices": [slice(0, 1), slice(0, 2), slice(0, 3)]}, [0, 1], [0, 2], [0, 3]],
        [
            {
                "resolution": [3, 5, 7],
                "slices": [slice(0, 1), slice(0, 2), slice(0, 3)],
            },
            [0, 3],
            [0, 10],
            [0, 21],
        ],
        [
            {"start_coord": [1, 2, 3], "end_coord": [11, 12, 13]},
            [1, 11],
            [2, 12],
            [3, 13],
        ],
        [
            {"start_coord": "1, 2, 3", "end_coord": "11, 12, 13"},
            [1, 11],
            [2, 12],
            [3, 13],
        ],
    ],
)
def test_constructor(kwargs, expected_x, expected_y, expected_z):
    result = BoundingCube(**kwargs)
    assert result.x_range == expected_x
    assert result.y_range == expected_y
    assert result.z_range == expected_z


@pytest.mark.parametrize(
    "kwargs",
    [
        {"slices": [slice(0, 1), slice(0, 2), slice(0, 3)], "start_coord": [1, 2, 3]},
        {
            "slices": [slice(0, 1), slice(0, 2), slice(0, 3)],
            "start_coord": [1, 2, 3],
            "end_coord": [1, 2, 3],
        },
        {"start_coord": [1, 2, 3]},
        {"end_coord": [1, 2, 3]},
    ],
)
def test_constructor_exc(kwargs):
    with pytest.raises(ValueError):
        BoundingCube(**kwargs)


@pytest.fixture
def empty_bcube():
    return


@pytest.mark.parametrize(
    "bcube, pad_kwargs, expected",
    [
        [
            BoundingCube(start_coord="0, 0, 0", end_coord="0, 0, 0"),
            {"x_pad": 1, "y_pad": 3, "z_pad": 5},
            BoundingCube(start_coord="-1, -3, -5", end_coord="1, 3, 5"),
        ],
        [
            BoundingCube(start_coord="-1, -9, -25", end_coord="1, 9, 25"),
        ],
    ],
)
def test_pad(bcube, pad_kwargs, expected):
    result = bcube.pad(**pad_kwargs)
    assert result == expected
