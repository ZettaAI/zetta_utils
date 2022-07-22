# pylint: disable=missing-docstring
import typing
import pytest
import zetta_utils as zu


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
    result = zu.typing.check_type(obj, cls)
    assert result == expected
