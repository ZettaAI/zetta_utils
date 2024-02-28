# pylint: disable=missing-docstring,invalid-name
import torch

from zetta_utils.tensor_ops import multitensor

from ..helpers import assert_array_equal


def test_skip_on_empty_datas(mocker):
    wrapie = mocker.MagicMock()

    def wrapie_call(data1: torch.Tensor, data2: torch.Tensor) -> torch.Tensor:
        return data1 + data2 + 1

    wrapie.__call__ = wrapie_call

    wrapped = multitensor.skip_on_empty_datas(wrapie)
    assert wrapped(torch.Tensor([0]), torch.Tensor([0])) == torch.Tensor([0])


def test_compute_pixel_error():
    a = torch.Tensor(
        [
            [
                [3, 3, 0, 0, 0],
                [3, 3, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 1, 1],
            ]
        ]
    ).unsqueeze(-1)

    b = torch.Tensor(
        [
            [
                [1, 1, 2, 3, 2],
                [1, 1, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 1, 1, 0],
                [0, 0, 1, 1, 1],
            ]
        ]
    ).unsqueeze(-1)

    expected = torch.Tensor(
        [
            [
                [2, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        ]
    ).unsqueeze(-1)

    result_torch = multitensor.compute_pixel_error(a, b, 3)
    assert_array_equal(result_torch, expected)
    result_np = multitensor.compute_pixel_error(a.numpy(), b.numpy(), 3)
    assert_array_equal(result_np, expected.numpy())


def test_erode_combine():
    a = torch.Tensor(
        [
            [
                [3, 3, 0, 0, 0],
                [3, 3, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 1, 5],
            ]
        ]
    ).unsqueeze(-1)

    b = torch.Tensor(
        [
            [
                [1, 1, 2, 3, 2],
                [1, 1, 0, 0, 0],
                [4, 4, 0, 0, 1],
                [0, 0, 1, 1, 0],
                [0, 0, 1, 1, 1],
            ]
        ]
    ).unsqueeze(-1)

    expected = torch.Tensor(
        [
            [
                [2, 1, 2, 3, 2],
                [1, 1, 0, 0, 0],
                [4, 4, 0, 0, 1],
                [0, 0, 1, 1, 0],
                [0, 0, 1, 1, 5],
            ]
        ]
    ).unsqueeze(-1)
    result_torch = multitensor.erode_combine(a, b, 3)
    assert_array_equal(result_torch, expected)
    result_np = multitensor.erode_combine(a.numpy(), b.numpy(), 3)
    assert_array_equal(result_np, expected.numpy())
