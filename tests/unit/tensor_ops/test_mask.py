# pylint: disable=missing-docstring,invalid-name
import torch

from zetta_utils.tensor_ops import mask
from ..helpers import assert_array_equal


def test_skip_on_empty_data():
    def wrapie(data: torch.Tensor) -> torch.Tensor:
        return data + 1

    wrapped = mask.skip_on_empty_data(wrapie)
    assert wrapped(torch.Tensor([0])) == torch.Tensor([0])


def test_filter_cc_small():
    a = torch.Tensor(
            [
                [
                    [1, 1, 0, 1],
                    [1, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 1, 1],
                ]
            ]
    ).unsqueeze(-1)

    expected = torch.Tensor(
            [
                [
                    [0, 0, 0, 1],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 1, 1],
                ]
            ]
    ).unsqueeze(-1)

    result = mask.filter_cc(
        a,
        mode="keep_small",
        thr=2,
    )
    assert_array_equal(result, expected)


def test_filter_cc_big():
    a = torch.Tensor(
        [
            [1, 1, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 1],
        ]
    )

    expected = torch.Tensor(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )

    result = mask.filter_cc(
        a,
        mode="keep_large",
        thr=2,
    )
    assert_array_equal(result, expected)


def test_coarsen_width1():
    a = (
        torch.Tensor(
                [
                    [
                        [1, 1, 0, 0],
                        [1, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 1],
                    ]
                ]
        ).unsqueeze(-1)
        > 0
    )

    expected = (
        torch.Tensor(
                [
                    [
                        [1, 1, 1, 0],
                        [1, 1, 1, 0],
                        [1, 1, 1, 1],
                        [0, 0, 1, 1],
                    ]
                ]
        ).unsqueeze(-1)
        > 0
    )

    result = mask.coarsen(
        a,
        width=1,
    )
    assert_array_equal(result, expected)


def test_binary_closing():
    a = torch.tensor(
            [
                [
                    [0, 0, 0, 0],
                    [0, 1, 1, 1],
                    [0, 1, 0, 1],
                    [0, 1, 1, 1],
                ]
            ]
    ).unsqueeze(-1)

    expected = torch.Tensor(
            [
                [
                    [0, 0, 0, 0],
                    [0, 1, 1, 1],
                    [0, 1, 1, 1],
                    [0, 1, 1, 1],
                ]
            ]
    ).unsqueeze(-1)

    result = mask.binary_closing(
        a,
    )
    assert_array_equal(result.bool(), expected.bool())
