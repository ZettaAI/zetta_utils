import typing

import pytest
import numpy as np
import torch

import zetta_utils as zu

from .. import helpers


# if you use ``attrs``, no need to test constructor

# use parametrize to test the same behaviors
@pytest.mark.parametrize(
    "field1, field2, data, expected",
    [
        [1, None, np.ones((2, 2)), np.ones((2, 2)) * 2],
        [2, 1.5, torch.ones((2, 2)), torch.ones((2, 2)) * 4],
    ],
)
def test_example_processor_mode1(
    field1: int, field2: typing.Optional[float], data: zu.typing.Tensor, expected: zu.typing.Tensor
):
    proc = zu.processors.zoo.example_processor.ExampleProcessor(field1=field1, field2=field2)
    result = proc(data)

    helpers.assert_array_equal(
        result, expected
    )  # use ``helpers.assert_array_equal`` to test equality of np.ndarray + torch.Tensor


# don't have to use parametrize when using jsut one testcase
def test_example_processor_mode2(mocker):
    expected = mocker.Mock()
    # Use mocker to patch code that's not yours.
    multiply_m = mocker.patch(
        "zetta_utils.tensor.ops.multiply",
        return_value=expected,  # Return mocks, as they always pass all type checks.
    )
    field1 = -1
    field2 = 2
    data = np.ones((2, 2))
    proc = zu.processors.zoo.example_processor.ExampleProcessor(field1=field1, field2=field2)
    result = proc(data)

    assert expected == result
    multiply_m.assert_called_with(data, field2)


# test exceptions
# Together, all your tests should cover 100% of your code
# Use ``mocker`` freely to make it as simple to test as possible
@pytest.mark.parametrize(
    "field1, field2, data, expected_exc",
    [
        [-1, None, np.ones((2, 2)), ValueError],
        [-2, None, torch.ones((2, 2)), ValueError],
    ],
)
def test_example_processor_exc(
    field1: int, field2: typing.Optional[float], data: zu.typing.Tensor, expected_exc
):
    proc = zu.processors.zoo.example_processor.ExampleProcessor(field1=field1, field2=field2)
    with pytest.raises(expected_exc):
        proc(data)
