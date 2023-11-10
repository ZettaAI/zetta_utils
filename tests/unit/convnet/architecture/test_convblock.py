# pylint: disable=protected-access
from __future__ import annotations

import pytest
import torch

from zetta_utils import convnet

from ...helpers import assert_array_equal


@pytest.mark.parametrize(
    "num_channels",
    [
        [1, 2],
        [1, 2, 7],
        [33, 44, 55, 77],
    ],
)
def test_channel_number(num_channels: list[int]):
    block = convnet.architecture.ConvBlock(num_channels=num_channels, kernel_sizes=[1, 2])
    conv_count = 0
    for e in block.layers:
        if isinstance(e, torch.nn.modules.conv._ConvNd):
            assert e.in_channels == num_channels[conv_count]
            assert e.out_channels == num_channels[conv_count + 1]
            conv_count += 1

    assert conv_count == len(num_channels) - 1


@pytest.mark.parametrize(
    "kernel_sizes, expected",
    [
        [[3, 3], [(3, 3), (3, 3)]],
        [[3, 5], [(3, 5), (3, 5)]],
        [(3, 1), [(3, 1), (3, 1)]],
        [[(3, 1), (3, 2)], [(3, 1), (3, 2)]],
    ],
)
def test_kernel_size(kernel_sizes, expected: list[tuple[int]]):
    block = convnet.architecture.ConvBlock(num_channels=[1, 2, 3], kernel_sizes=kernel_sizes)

    conv_count = 0
    for e in block.layers:
        if isinstance(e, torch.nn.modules.conv._ConvNd):
            assert e.kernel_size == expected[conv_count]
            conv_count += 1


@pytest.mark.parametrize(
    "paddings, expected",
    [
        [[3, 3], [(3, 3), (3, 3)]],
        [[3, 5], [(3, 5), (3, 5)]],
        [(3, 1), [(3, 1), (3, 1)]],
        [[(3, 1), (3, 2)], [(3, 1), (3, 2)]],
    ],
)
def test_padding(paddings, expected: list[tuple[int]]):
    block = convnet.architecture.ConvBlock(
        num_channels=[1, 2, 3], kernel_sizes=[3, 3], paddings=paddings
    )

    conv_count = 0
    for e in block.layers:
        if isinstance(e, torch.nn.modules.conv._ConvNd):
            assert e.padding == expected[conv_count]
            conv_count += 1


def test_length_mismatch():
    with pytest.raises(ValueError):
        convnet.architecture.ConvBlock(
            num_channels=[1, 2, 3], paddings=[[2, 2, 2]], kernel_sizes=[1, 2]
        )


def test_norm():
    block = convnet.architecture.convblock.ConvBlock(
        num_channels=[1, 2, 3], normalization=torch.nn.BatchNorm2d, kernel_sizes=[2, 2]
    )
    norm_count = 0
    for e in block.layers:
        if isinstance(e, torch.nn.BatchNorm2d):
            norm_count += 1
    assert norm_count == 1


def test_norm_last():
    block = convnet.architecture.ConvBlock(
        num_channels=[1, 2, 3],
        kernel_sizes=[3, 3],
        normalization=torch.nn.BatchNorm2d,
        normalize_last=True,
    )
    norm_count = 0
    for e in block.layers:
        if isinstance(e, torch.nn.BatchNorm2d):
            norm_count += 1
    assert norm_count == 2


def not_test_forward_naive(mocker):
    mocker.patch("torch.nn.Conv2d.forward", lambda _, x: x)
    block = convnet.architecture.ConvBlock(num_channels=[1, 2, 3, 4, 5], kernel_sizes=[3, 3])
    result = block(torch.zeros([1, 1, 1, 1]))
    assert_array_equal(
        result.cpu().detach().numpy(), torch.zeros([1, 1, 1, 1]).cpu().detach().numpy()
    )


@pytest.mark.parametrize(
    "skips, expected",
    [
        # fmt: off
        [
            None,
            1 * 2 * 2 * 2 * 2           # 4 convolutions
        ],
        [
            {"0": 2, "1": 2, "2": 3},
            (((1 * 2 * 2)               # first 2 convolutions
            + 1                         # Skip content "0": 2
            + 1 * 2                     # Skip content "1": 2
            ) * 2                       # third convolution
            + (1 * 2 * 2)               # Skip content "2": 3
            + 1                         # also includes the skip content "0": 2
            + 1 * 2                     # also includes the skip content "1": 2
            ) * 2                       # fourth convolution
        ],
        [
            {"0": 3, "1": 3, "2": 4},
            ((1 * 2 * 2 * 2)            # first 3 convolutions
            + 1                         # Skip content "0": 3
            + 1 * 2                     # Skip content "1": 3
            ) * 2                       # fourth convolution
            + (1 * 2 * 2)               # Skip content "2": 4
        ],
        # fmt: on
    ],
)
def test_forward_skips(mocker, skips, expected):
    mocker.patch("torch.nn.Conv2d.forward", lambda _, x: 2 * x)
    block = convnet.architecture.ConvBlock(kernel_sizes=[1], num_channels=[1] * 5, skips=skips)
    result = block(torch.ones((1, 1, 1, 1)))
    assert_array_equal(result.detach().numpy(), torch.full((1, 1, 1, 1), expected).numpy())
