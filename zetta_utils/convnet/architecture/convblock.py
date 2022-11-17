# pylint: disable=protected-access
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple, Union

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
    :param kernel_sizes: Convolution kernel sizes. When specified as a single integer or a
        tuple, it will be passed as ``k`` parameter to all convolution constructors.
        When specified as a list, each item in the list will be passed as ``k`` to the
        corresponding convolution in order. The list length must match the number of
        convolutions.
    :param strides: Convolution strides. When specified as a single integer or a
        tuple, it will be passed as the stride parameter to all convolution constructors.
        When specified as a list, each item in the list will be passed as stride to the
        corresponding convolution in order. The list length must match the number of
        convolutions.
    :param paddings: Convolution padding sizes. When specified as a single integer or a
        tuple, it will be passed as the padding parameter to all convolution constructors.
        When specified as a list, each item in the list will be passed as padding to the
        corresponding convolution in order. The list length must match the number of
        convolutions.
    :param skips: Specification for residual skip connection. For example,
        ``skips={"1": 3}`` specifies a single residual skip connection from the output of the
        first convolution (index 1) to the third covnolution (index 3). 0 specifies the input
        to the first layer.
    :param normalize_last: Whether to apply normalization after the last layer.
    :param activate_last: Whether to apply activation after the last layer.
    """

    def __init__(  # pylint: disable=too-many-locals
        self,
        num_channels: List[int],
        activation: Callable[[], torch.nn.Module] = torch.nn.LeakyReLU,
        conv: Callable[..., torch.nn.modules.conv._ConvNd] = torch.nn.Conv2d,
        normalization: Optional[Callable[[int], torch.nn.Module]] = None,
        kernel_sizes: Union[List[int], int, List[Tuple[int, ...]], Tuple[int, ...]] = 3,
        strides: Union[List[int], int, List[Tuple[int, ...]], Tuple[int, ...]] = 1,
        paddings: Union[List[int], int, List[Tuple[int, ...]], Tuple[int, ...]] = 1,
        skips: Optional[Dict[Union[int, str], int]] = None,
        normalize_last: bool = False,
        activate_last: bool = False,
    ):
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

        if isinstance(strides, list):
            strides_ = strides  # type: Union[List[int], List[Tuple[int, ...]]]
        else:
            strides_ = [strides for _ in range(len(num_channels) - 1)]  # type: ignore

        assert len(strides_) == (len(num_channels) - 1)

        if isinstance(paddings, list):
            paddings_ = paddings  # type: Union[List[int], List[Tuple[int, ...]]]
        else:
            paddings_ = [paddings for _ in range(len(num_channels) - 1)]  # type: ignore

        assert len(paddings_) == (len(num_channels) - 1)

        for i, (ch_in, ch_out) in enumerate(zip(num_channels[:-1], num_channels[1:])):
            self.layers.append(conv(ch_in, ch_out, kernel_sizes_[i], strides_[i], paddings_[i]))

            is_last_conv = i == len(num_channels) - 2

            if (normalization is not None) and ((not is_last_conv) or normalize_last):
                self.layers.append(normalization(ch_out))  # pylint: disable=not-callable

            if (not is_last_conv) or activate_last:
                self.layers.append(activation())

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        skip_data_for = defaultdict(
            lambda: torch.zeros_like(data, device=data.device)
        )  # type: Dict[int, torch.Tensor]
        conv_count = 1
        if 0 in self.skips:
            skip_dest = self.skips[0]
            skip_data_for[skip_dest] += data
        result = data
        for this_layer, next_layer in zip(self.layers, self.layers[1:] + [None]):
            if isinstance(this_layer, torch.nn.modules.conv._ConvNd):
                if conv_count in skip_data_for:
                    result += skip_data_for[conv_count]

            # breakpoint()
            result = this_layer(result)

            if isinstance(next_layer, torch.nn.modules.conv._ConvNd):
                if conv_count in self.skips:
                    skip_dest = self.skips[conv_count]
                    skip_data_for[skip_dest] += result

                conv_count += 1

        return result
