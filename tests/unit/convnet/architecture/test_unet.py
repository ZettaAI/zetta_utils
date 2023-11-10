# pylint: disable=protected-access
from __future__ import annotations

from functools import partial

import pytest
import torch

from zetta_utils import convnet

from ...helpers import assert_array_equal


@pytest.mark.parametrize(
    "list_num_channels",
    [
        [[1, 2, 2], [2, 2], [2, 2, 1]],
        [[1, 2, 7], [7, 3, 7], [7, 5, 3]],
        [
            [33, 44, 55, 77],
            [77, 22, 11],
            [11, 1, 2, 11],
            [11, 9, 2, 3, 5, 7, 77],
            [77, 31, 61, 51, 33],
        ],
    ],
)
def test_channel_number(list_num_channels: list[list[int]]):
    unet = convnet.architecture.UNet(
        kernel_sizes=[3, 3],
        list_num_channels=list_num_channels,
        downsample=partial(torch.nn.AvgPool2d, kernel_size=2),
        upsample=partial(torch.nn.Upsample, scale_factor=2),
    )
    in_num_channels = [c for block in list_num_channels for c in block[:-1]]
    out_num_channels = [c for block in list_num_channels for c in block[1:]]
    conv_count = 0
    for f in unet.layers:
        if hasattr(f, "layers"):
            for e in f.layers:
                if isinstance(e, torch.nn.modules.conv._ConvNd):
                    assert e.in_channels == in_num_channels[conv_count]
                    assert e.out_channels == out_num_channels[conv_count]
                    conv_count += 1

    assert conv_count == len(in_num_channels)


@pytest.mark.parametrize(
    "kernel_sizes, expected",
    [
        [(3, 3), [(3, 3), (3, 3), (3, 3)]],
        [(3, 1), [(3, 1), (3, 1), (3, 1)]],
    ],
)
def test_kernel_size(kernel_sizes, expected: list[tuple[int]]):
    unet = convnet.architecture.UNet(
        list_num_channels=[[1, 3], [3, 3], [3, 1]],
        kernel_sizes=kernel_sizes,
        downsample=partial(torch.nn.AvgPool2d, kernel_size=2),
        upsample=partial(torch.nn.Upsample, scale_factor=2),
    )
    conv_count = 0
    for f in unet.layers:
        if hasattr(f, "layers"):
            for e in f.layers:
                if isinstance(e, torch.nn.modules.conv._ConvNd):
                    assert e.kernel_size == expected[conv_count]
                    conv_count += 1


@pytest.mark.parametrize(
    "strides, expected",
    [
        [(3, 3), [(3, 3), (3, 3), (3, 3)]],
        [(3, 1), [(3, 1), (3, 1), (3, 1)]],
    ],
)
def test_stride(strides, expected: list[tuple[int]]):
    unet = convnet.architecture.UNet(
        kernel_sizes=[3, 3],
        list_num_channels=[[1, 3], [3, 3], [3, 1]],
        strides=strides,
        paddings=[1, 1],
        downsample=partial(torch.nn.Conv2d, kernel_size=2, stride=2, padding=0),
        upsample=partial(torch.nn.Upsample, scale_factor=2),
    )
    conv_count = 0
    for f in unet.layers:
        if hasattr(f, "layers"):
            for e in f.layers:
                if isinstance(e, torch.nn.modules.conv._ConvNd):
                    assert e.stride == expected[conv_count]
                    conv_count += 1


@pytest.mark.parametrize(
    "paddings, expected",
    [
        [(3, 3), [(3, 3), (3, 3), (3, 3)]],
        [(3, 1), [(3, 1), (3, 1), (3, 1)]],
    ],
)
def test_padding(paddings, expected: list[tuple[int]]):
    unet = convnet.architecture.UNet(
        list_num_channels=[[1, 3], [3, 3], [3, 1]],
        kernel_sizes=[3, 3],
        paddings=paddings,
        downsample=partial(torch.nn.AvgPool2d, kernel_size=2),
        upsample=partial(torch.nn.Upsample, scale_factor=2),
    )
    conv_count = 0
    for f in unet.layers:
        if hasattr(f, "layers"):
            for e in f.layers:
                if isinstance(e, torch.nn.modules.conv._ConvNd):
                    assert e.padding == expected[conv_count]
                    conv_count += 1


def test_norm():
    unet = convnet.architecture.UNet(
        kernel_sizes=[3, 3],
        list_num_channels=[[1, 3], [3, 3], [3, 1]],
        normalization=torch.nn.BatchNorm2d,
        downsample=partial(torch.nn.AvgPool2d, kernel_size=2),
        upsample=partial(torch.nn.ConvTranspose2d, kernel_size=2, stride=2, padding=0),
    )
    norm_count = 0
    for f in unet.layers:
        if hasattr(f, "layers"):
            for e in f.layers:
                if isinstance(e, torch.nn.BatchNorm2d):
                    norm_count += 1
    assert norm_count == 2


def test_norm_last():
    unet = convnet.architecture.UNet(
        kernel_sizes=[3, 3],
        list_num_channels=[[1, 3], [3, 3], [3, 1]],
        normalization=torch.nn.BatchNorm2d,
        downsample=partial(torch.nn.AvgPool2d, kernel_size=2),
        upsample=partial(torch.nn.ConvTranspose2d, kernel_size=2, stride=2, padding=0),
        normalize_last=True,
    )
    norm_count = 0
    for f in unet.layers:
        if hasattr(f, "layers"):
            for e in f.layers:
                if isinstance(e, torch.nn.BatchNorm2d):
                    norm_count += 1
    assert norm_count == 3


def test_activate_last():
    unet = convnet.architecture.UNet(
        kernel_sizes=[3, 3],
        list_num_channels=[[1, 3], [3, 3], [3, 1]],
        normalization=torch.nn.BatchNorm2d,
        downsample=partial(torch.nn.AvgPool2d, kernel_size=2),
        upsample=partial(torch.nn.ConvTranspose2d, kernel_size=2, stride=2, padding=0),
        activate_last=True,
    )
    act_count = 0
    for f in unet.layers:
        if hasattr(f, "layers"):
            for e in f.layers:
                if isinstance(e, torch.nn.LeakyReLU):
                    act_count += 1
    assert act_count == 3


def not_test_forward_naive(mocker):
    mocker.patch("torch.nn.Conv2d.forward", lambda _, x: x)
    unet = convnet.architecture.UNet(
        kernel_sizes=[3, 3],
        list_num_channels=[[1, 1], [1, 1], [1, 1]],
        downsample=partial(torch.nn.AvgPool2d, kernel_size=2),
        upsample=partial(torch.nn.Upsample, scale_factor=2),
    )
    result = unet.forward(torch.ones([1, 1, 4, 4]))
    assert_array_equal(
        result.cpu().detach().numpy(), 2 * torch.ones([1, 1, 4, 4]).cpu().detach().numpy()
    )


@pytest.mark.parametrize(
    "skips, expected",
    [
        # fmt: off
        [
            None,
            (1 * 2 * 2          # L0 convblock, left
            * 2 * 2 * 2         # L1 convblock
            + 1 * 2 * 2         # UNet skip connection
            ) * 2 * 2           # L0 convblock, right
        ],
        [
            {"0": 1},
            ((((1 * 2 + 1) * 2  # L0 convblock, left   --> 6
            * 2 + 6) * 2 * 2    # L1 convblock         --> 72
            + 6                 # UNet skip connection --> 78
            ) * 2 + 78) * 2     # L0 convblock, right  --> 468
        ],
        [
            {"0": 2},
            ((((1 * 2 * 2 + 1)  # L0 convblock, left   --> 5
            * 2 * 2 + 5) * 2    # L1 convblock         --> 50
            + 5                 # UNet skip connection --> 55
            ) * 2 * 2 + 55)     # L0 convblock, right  --> 275
        ],
        # fmt: on
    ],
)
def test_forward_skips(mocker, skips, expected):
    mocker.patch("torch.nn.Conv2d.forward", lambda _, x: 2 * x)
    unet = convnet.architecture.UNet(
        kernel_sizes=[1],
        list_num_channels=[[1, 1, 1], [1, 1, 1, 1], [1, 1, 1]],
        downsample=partial(torch.nn.AvgPool2d, kernel_size=1),
        upsample=partial(torch.nn.Upsample, scale_factor=1),
        skips=skips,
        unet_skip_mode="sum",
    )
    result = unet.forward(torch.ones((1, 1, 1, 1)))
    assert_array_equal(result.detach().numpy(), torch.full((1, 1, 1, 1), expected).numpy())


@pytest.mark.parametrize(
    "unet_skip_mode, expected_result",
    [
        ["sum", torch.tensor([[[[2.0, 2.0], [2.0, 2.0]]]])],
        ["concat", torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]])],
    ],
)
def test_skip_mode(unet_skip_mode, expected_result, mocker):
    conv2d_forward = mocker.patch("torch.nn.Conv2d.forward")
    conv2d_forward.side_effect = lambda x: torch.prod(torch.Tensor(x), dim=1, keepdim=True)
    unet = convnet.architecture.UNet(
        kernel_sizes=[3, 3],
        list_num_channels=[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        downsample=partial(torch.nn.AvgPool2d, kernel_size=2),
        upsample=partial(torch.nn.Upsample, scale_factor=2),
        unet_skip_mode=unet_skip_mode,
    )
    result = unet.forward(torch.ones([1, 1, 2, 2]))

    assert_array_equal(result.detach().cpu().numpy(), expected_result.detach().cpu().numpy())
