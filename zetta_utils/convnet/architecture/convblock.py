# pylint: disable=protected-access
from __future__ import annotations

from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
from torch import nn
from typeguard import typechecked

from zetta_utils import builder


@builder.register("ConvBlock")
@typechecked
# cant use attrs because torch Module + attrs combination makes pylint freak out
class ConvBlock(nn.Module):
    """
    A block with sequential convolutions with activations, optional normalization and residual
    connections.

    :param num_channels: List of integers specifying the number of channels of each
        convolution. For example, specification [1, 2, 3] will correspond to a sequence of
        2 convolutions, where the first one has 1 input channel and two output channels and
        the second one has 2 input channels and 3 output channels.
    :param conv: Constructor for convolution layers.
    :param activation: Constructor for activation layers.
    :param normalization: Constructor for normalization layers. Normalization will
        be applied after convolution before activation.
    :param padding_mode: Convolution padding mode.
    :param kernel_sizes: Convolution kernel sizes. When specified as a single integer or a
        tuple, it will be passes as ``k`` parameter to all convolution constructors.
        When specified as a list, each item in the list will be passed as ``k`` to the
        corresponding convolution in order. The list length must match the number of
        convolutions.
    :param skips: Specification for residual skip connection. For example,
        ``skips={"0": 2}`` specifies a single residual skip connection from the output of the
        first convolution (index 0) to the third covnolution (index 2).
    :param normalize_last: Whether to apply normalization after the last layer.
    :param activate_last: Whether to apply activation after the last layer.
    """

    def __init__(
        self,
        num_channels: List[int],
        activation: Callable[[], torch.nn.Module] = torch.nn.LeakyReLU,
        conv: Callable[..., torch.nn.modules.conv._ConvNd] = torch.nn.Conv2d,
        padding_mode: Union[Literal["valid"], Literal["same"]] = "same",
        normalization: Optional[Callable[[int], torch.nn.Module]] = None,
        kernel_sizes: Union[List[int], int, List[Tuple[int, ...]], Tuple[int, ...]] = 3,
        skips: Optional[Dict[Union[int, str], int]] = None,
        normalize_last: bool = False,
        activate_last: bool = False,
    ):  # pylint: disable=too-many-locals
        super().__init__()
        if skips is None:
            self.skips = {}
        else:
            self.skips = {int(k): v for k, v in skips.items()}
        self.layers = torch.nn.ModuleList()

        if isinstance(kernel_sizes, list):
            kernel_sizes_ = kernel_sizes  # type: Union[List[int], List[Tuple[int, ...]]]
        else:
            kernel_sizes_ = [kernel_sizes for _ in range(len(num_channels) - 1)]  # type: ignore

        assert len(kernel_sizes_) == (len(num_channels) - 1)

        for i, (ch_in, ch_out) in enumerate(zip(num_channels[:-1], num_channels[1:])):
            new_conv = conv(ch_in, ch_out, kernel_sizes_[i], padding=padding_mode)
            # TODO: make this step optional
            if not new_conv.bias is None:
                new_conv.bias.data[:] = 0
            self.layers.append(new_conv)

            is_last_conv = i == len(num_channels) - 2

            if (normalization is not None) and ((not is_last_conv) or normalize_last):
                self.layers.append(normalization(ch_out))  # pylint: disable=not-callable

            if (not is_last_conv) or activate_last:
                self.layers.append(activation())

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        skip_data_for = {}  # type: Dict[int, torch.Tensor]
        conv_count = 0

        result = data
        for this_layer, next_layer in zip(self.layers, list(self.layers[1:]) + [None]):
            if isinstance(this_layer, torch.nn.modules.conv._ConvNd):
                if conv_count in skip_data_for:
                    result += skip_data_for[conv_count]

            result = this_layer(result)

            if isinstance(next_layer, torch.nn.modules.conv._ConvNd):
                if conv_count in self.skips:
                    skip_dest = self.skips[conv_count]
                    if skip_dest in skip_data_for:
                        skip_data_for[skip_dest] += result
                    else:
                        skip_data_for[skip_dest] = result

                conv_count += 1

        return result
