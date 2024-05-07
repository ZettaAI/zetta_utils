# pylint: disable=missing-docstring
import numpy as np
import pytest
import torch

from zetta_utils.tensor_ops.normalization import apply_clahe

device = "cuda" if torch.cuda.is_available() else "cpu"  # pylint: disable=invalid-name


@pytest.mark.parametrize(
    "in_tensor, expected_shape, on_cuda",
    [
        [np.ones((32, 64), dtype=np.uint8), (32, 64), False],
        [torch.zeros((64, 64), dtype=torch.int8, device="cpu"), (64, 64), False],
        [
            torch.ones((1, 1, 37, 42), dtype=torch.int8, device=device),
            (1, 1, 37, 42),
            device == "cuda",
        ],
    ],
)
def test_result_matches_input(in_tensor, expected_shape, on_cuda):
    result = apply_clahe(in_tensor)
    assert result.shape == expected_shape
    assert result.is_cuda == on_cuda


def test_int8_and_uint8():
    int8_tensor = torch.full((64, 64), 42, dtype=torch.int8)
    uint8_tensor = torch.full((64, 64), 42 + 128, dtype=torch.uint8)
    assert torch.equal(apply_clahe(int8_tensor), apply_clahe(uint8_tensor))


@pytest.mark.parametrize(
    "in_tensor",
    [
        np.ones((123, 456), dtype=np.float32),
        torch.zeros((98, 76), dtype=torch.int16, device="cpu"),
    ],
)
def test_non_int8_or_uint8_exc(in_tensor):
    with pytest.raises(NotImplementedError):
        apply_clahe(in_tensor)


@pytest.mark.parametrize(
    "in_tensor",
    [np.ones((2, 32, 64), dtype=np.uint8), torch.zeros((64), dtype=torch.int8, device="cpu")],
)
def test_non2d_exc(in_tensor):
    with pytest.raises(NotImplementedError):
        apply_clahe(in_tensor)
