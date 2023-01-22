# pylint: disable=all
import typing

import numpy as np
import pytest

import zetta_utils.typing as zt


@pytest.mark.parametrize(
    "obj, cls, expected",
    [
        [1, int, True],
        ["hello", typing.Literal["hello"], True],
        ["goodbye", typing.Union[typing.Literal["hello"], str], True],
        ["goodbye", typing.Union[typing.Literal["hello"], int], False],
    ],
)
def test_check_type(obj, cls, expected):
    result = zt.check_type(obj, cls)
    assert result == expected
