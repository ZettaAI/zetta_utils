# pylint: disable=missing-docstring
import pytest
from zetta_utils import distributions


@pytest.mark.parametrize(
    "x, expected",
    [
        [100, distributions.uniform_dist(100, 100)],
        [1.5, distributions.uniform_dist(1.5, 1.5)],
        [distributions.uniform_dist(1.0, 1.5), distributions.uniform_dist(1.0, 1.5)],
    ],
)
def test_ensure_distribution(x, expected):
    result = distributions.to_distribution(x)
    assert result == expected
