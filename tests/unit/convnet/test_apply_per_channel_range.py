# pylint: disable=protected-access
import functools
import pytest
import torch

from zetta_utils import convnet, tensor_ops
from ..helpers import assert_array_equal


def test_forward():
    appp = convnet.pieces.ApplyPerChannelRange(apply_fns=[functools.partial(tensor_ops)])

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
        [3, [(3, 3), (3, 3)]],
        [[3, 5], [(3, 3), (5, 5)]],
        [(3, 1), [(3, 1), (3, 1)]],
        [[(3, 1), (3, 2)], [(3, 1), (3, 2)]],
    ],
)
def test_kernel_size(kernel_sizes, expected: list[tuple[int]]):
    block = convnet.pieces.ConvBlock(num_channels=[1, 2, 3], kernel_sizes=kernel_sizes)

    conv_count = 0
    for e in block.layers:
        if isinstance(e, torch.nn.modules.conv._ConvNd):
            assert e.kernel_size == expected[conv_count]
            conv_count += 1


def test_norm():
    block = convnet.pieces.convblock.ConvBlock(
        num_channels=[1, 2, 3], normalization=torch.nn.BatchNorm2d
    )
    norm_count = 0
    for e in block.layers:
        if isinstance(e, torch.nn.BatchNorm2d):
            norm_count += 1
    assert norm_count == 1


def test_norm_last():
    block = convnet.pieces.ConvBlock(
        num_channels=[1, 2, 3], normalization=torch.nn.BatchNorm2d, normalize_last=True
    )
    norm_count = 0
    for e in block.layers:
        if isinstance(e, torch.nn.BatchNorm2d):
            norm_count += 1
    assert norm_count == 2


def not_test_forward_naive(mocker):
    mocker.patch("torch.nn.Conv2d.forward", lambda _, x: x)
    block = convnet.pieces.ConvBlock(num_channels=[1, 2, 3, 4, 5])
    result = block(torch.zeros([1, 1, 1, 1]))
    assert_array_equal(
        result.cpu().detach().numpy(), torch.zeros([1, 1, 1, 1]).cpu().detach().numpy()
    )


def test_forward_skips(mocker):
    mocker.patch("torch.nn.Conv2d.forward", lambda _, x: x)
    block = convnet.pieces.ConvBlock(num_channels=[1, 2, 3, 4, 5], skips={"0": 2, "1": 2, "2": 3})
    result = block(torch.ones([1, 1, 1, 1]))
    assert_array_equal(
        result.cpu().detach().numpy(), 6 * torch.ones([1, 1, 1, 1]).cpu().detach().numpy()
    )
