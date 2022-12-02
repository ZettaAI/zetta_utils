# pylint: disable=protected-access
from __future__ import annotations

from functools import partial
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
from torch import nn
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.convnet.architecture import ConvBlock

Padding = Union[Literal["same", "valid"], int, Tuple[int, ...]]


@builder.register("UNet")
@typechecked
class UNet(nn.Module):
    """
    A basic UNet that uses ConvBlocks. Note that the downsamples are applied after the regular
    convolutions, while the upsampleolutions are applied before the regular convolutions

    :param list_num_channels: List of lists of integers specifying the number of channels of
        each convolution. There must be an odd number of lists, for going up and going down,
        and the middle. Each list is used to generate the ConvBlocks.
    :param conv: Constructor for convolution layers.
    :param downsample: Constructor for downsample layers. Must include strides and kernel sizes.
        Activation and normalization are included by default and cannot be turned off.
    :param upsample: Constructor for upsample layers. Must include strides and kernel sizes.
        Activation and normalization are included by default and cannot be turned off.
    :param activation: Constructor for activation layers.
    :param normalization: Constructor for normalization layers. Normalization will
        be applied after convolution before activation.
    :param kernel_sizes: List of convolution kernel sizes. When specified as a single
        integer or a tuple, it will be passed as ``k`` parameter to all convolution constructors
        for all ConvBlocks.
        When specified as a list containing lists or a single or a tuple, each value will
        be passed as the ``kernel_size`` for each ConvBlock, with the behaviour documented in
        ConvBlock.
        The top level list length must match the number of ConvBlocks.
    :param strides: List of convolution strides. When specified as a single integer or a
        tuple, it will be passed as the stride parameter to all convolution constructors for
        all ConvBlocks.
        When specified as a list containing lists or a single or a tuple, each value will
        be passed for each ConvBlock as the ``strides`` parameter, with the behaviour documented
        in ConvBlock. The top level list length must match the number of ConvBlocks.
    :param paddings: List of convolution padding sizes. When specified as a single "same", "valid",
        integer or a tuple, it will be passed as the ``paddings`` parameter to all convolution
        constructors for all ConvBlocks.
        When specified as a list containing lists or a single or a tuple, each value will
        be passed for each ConvBlock as the ``paddings`` parameter, with the behaviour documented
        in ConvBlock. The top level list length must match the number of ConvBlocks.
    :param skips: List of specifications for residual skip connections as documented in ConvBlock.
        If nonempty, the list must match the number of ConvBlocks; this parameter will not be
        expanded.
    :param normalize_last: Whether to apply normalization after the last layer. Note that
        normalization is included by default within and at the end of the convblocks when the
        normalization layer is specified, and cannot be turned off.
    :param activate_last: Whether to apply activation after the last layer. Note that activation
        is included by default within and at the end of the convblocks and cannot be turned off.
    """

    def __init__(
        self,
        list_num_channels: List[List[int]],
        conv: Callable[..., torch.nn.modules.conv._ConvNd] = torch.nn.Conv2d,
        downsample: Callable[..., torch.nn.Module] = torch.nn.AvgPool2d,
        upsample: Callable[..., torch.nn.Module] = partial(torch.nn.Upsample, scale_factor=2),
        activation: Callable[[], torch.nn.Module] = torch.nn.LeakyReLU,
        normalization: Optional[Callable[[int], torch.nn.Module]] = None,
        kernel_sizes: Union[
            int,
            Tuple[int, ...],
            List[Union[int, Tuple[int, ...], List[Union[int, Tuple[int, ...]]]]],
        ] = 3,
        strides: Union[
            int,
            Tuple[int, ...],
            List[Union[int, Tuple[int, ...], List[Union[int, Tuple[int, ...]]]]],
        ] = 1,
        paddings: Union[Padding, List[Union[Padding, List[Padding]]]] = "same",
        skips: Optional[List[Dict[Union[int, str], int]]] = None,
        normalize_last: bool = False,
        activate_last: bool = False,
    ):  # pylint: disable=too-many-locals, too-many-statements, too-many-branches
        super().__init__()
        assert len(list_num_channels) % 2 == 1
        assert downsample is not None
        assert upsample is not None

        self.layers = torch.nn.ModuleList()

        if isinstance(kernel_sizes, list):
            kernel_sizes_ = (
                kernel_sizes
            )  # type: List[Union[int, Tuple[int, ...], List[Union[int, Tuple[int, ...]]]]]
            assert len(kernel_sizes_) == len(list_num_channels)
        else:
            kernel_sizes_ = [kernel_sizes for _ in range(len(list_num_channels))]

        if isinstance(strides, list):
            strides_ = (
                strides
            )  # type: List[Union[int, Tuple[int, ...], List[Union[int, Tuple[int, ...]]]]]
            assert len(strides_) == len(list_num_channels)
        else:
            strides_ = [strides for _ in range(len(list_num_channels))]

        if isinstance(paddings, list):
            paddings_ = paddings  # type: List[Union[Padding, List[Padding]]]
            assert len(paddings_) == len(list_num_channels)
        else:
            paddings_ = [paddings for _ in range(len(list_num_channels))]

        if isinstance(skips, list):
            skips_ = skips  # type: List[Dict[Union[int, str], int]]
            assert len(skips_) == len(list_num_channels)
        else:
            skips_ = [{} for _ in range(len(list_num_channels))]

        normalize_last_ = [(normalization is not None) for _ in range(len(list_num_channels))]
        normalize_last_[-1] = normalize_last
        activate_last_ = [True for _ in range(len(list_num_channels))]
        activate_last_[-1] = activate_last

        skips_in = []
        skips_out = []

        for i, num_channels in enumerate(list_num_channels):
            if i >= len(list_num_channels) // 2 + 1:
                skips_out.append(len(self.layers))
                try:
                    self.layers.append(
                        upsample(list_num_channels[i - 1][-1], list_num_channels[i][0])
                    )
                except:  # pylint: disable=bare-except
                    self.layers.append(upsample())
                if normalization is not None:
                    self.layers.append(normalization(list_num_channels[i][0]))
                self.layers.append(activation())

            self.layers.append(
                ConvBlock(
                    num_channels,
                    activation,
                    conv,
                    normalization,
                    kernel_sizes_[i],
                    strides_[i],
                    paddings_[i],
                    skips_[i],
                    normalize_last_[i],
                    activate_last_[i],
                )
            )

            if i < len(list_num_channels) // 2:
                try:
                    self.layers.append(
                        downsample(list_num_channels[i][-1], list_num_channels[i + 1][0])
                    )
                except:  # pylint: disable=bare-except
                    self.layers.append(downsample())
                if normalization is not None:
                    self.layers.append(normalization(list_num_channels[i + 1][0]))
                self.layers.append(activation())
                skips_in.append(len(self.layers) - 1)

        self.skips = {}
        for skip_in, skip_out in zip(skips_in, skips_out[-1::-1]):
            self.skips[skip_in] = skip_out

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        skip_data_for = {}  # type: Dict[int, torch.Tensor]
        result = data

        for i, layer in enumerate(self.layers):
            if i in skip_data_for:
                result += skip_data_for[i]

            result = layer(result)

            if i in self.skips:
                skip_data_for[self.skips[i]] = result

        return result
