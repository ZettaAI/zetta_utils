# type: ignore
# pylint: disable=protected-access, function-redefined
from __future__ import annotations

from typing import Callable, Literal, Sequence, Union

import torch
from torch import nn
from typeguard import typechecked
from typing_extensions import TypeAlias

from zetta_utils import builder
from zetta_utils.typing import ensure_seq_of_seq

Padding: TypeAlias = Union[Literal["same", "valid"], Sequence[int]]
PaddingMode: TypeAlias = Literal["zeros", "reflect", "replicate", "circular"]


@builder.register("ConvBlock", versions="<=0.0.1")
@typechecked
# cant use attrs because torch Module + attrs combination makes pylint freak out
class ConvBlock(nn.Module):
    """
    A block with sequential convolutions with activations, optional normalization and residual
    connections.
    :param num_channels: list of integers specifying the number of channels of each
        convolution. For example, specification [1, 2, 3] will correspond to a sequence of
        2 convolutions, where the first one has 1 input channel and 2 output channels and
        the second one has 2 input channels and 3 output channels.
    :param conv: Constructor for convolution layers.
    :param activation: Constructor for activation layers.
    :param normalization: Constructor for normalization layers. Normalization will
        be applied after convolution before activation.
    :param kernel_sizes: Convolution kernel sizes. When specified as a sequence of integerss
        it will be passed as ``k`` parameter to all convolution constructors.
        When specified as a sequence of sequence of integers, each item in the outer sequence will
        be passed as ``k`` to the corresponding convolution in order, and the outer sequence
        length must match the number of convolutions.
    :param strides: Convolution strides. When specified as a sequence of integers
        it will be passed as the stride parameter to all convolution constructors.
        When specified as a sequence of sequences, each item in the outer sequence will be passed
        as stride to the corresponding convolution in order, and the outer sequence length must
        match the number of convolutions.
    :param paddings: Convolution padding sizes.  When specified as a single string, or a
        sequence of integers, it will be passed as the padding parameter to all convolution
        constructors. When specified as a sequence of strings or sequence of int sequences, each
        item in the outer sequence will be passed as padding to the corresponding convolution in
        order, and the outer sequence length must match the number of convolutions.
    :param skips: Specification for residual skip connection. For example,
        ``skips={"1": 3}`` specifies a single residual skip connection from the output of the
        first convolution (index 1) to the input of third convolution (index 3).
        0 specifies the input to the first layer.
    :param normalize_last: Whether to apply normalization after the last layer.
    :param activate_last: Whether to apply activation after the last layer.
    :param padding_modes: Convolution padding modes. When specified as a single string,
        it will be passed as the padding mode parameter to all convolution constructors.
        When specified as a sequence of strings, each item in the sequence will be passed as
        padding mode to the corresponding convolution in order. The list length must match
        the number of convolutions.
    """

    def __init__(
        self,
        num_channels: Sequence[int],
        kernel_sizes: Sequence[int] | Sequence[Sequence[int]],
        conv: Callable[..., torch.nn.modules.conv._ConvNd] = torch.nn.Conv2d,
        strides: Sequence[int] | Sequence[Sequence[int]] | None = None,
        activation: Callable[[], torch.nn.Module] = torch.nn.LeakyReLU,
        normalization: Callable[[int], torch.nn.Module] | None = None,
        skips: dict[str, int] | None = None,
        normalize_last: bool = False,
        activate_last: bool = False,
        paddings: Padding | Sequence[Padding] = "same",
        padding_modes: PaddingMode | Sequence[PaddingMode] = "zeros",
    ):  # pylint: disable=too-many-locals
        super().__init__()
        if skips is None:
            self.skips = {}
        else:
            self.skips = skips

        self.layers = torch.nn.ModuleList()

        num_conv = len(num_channels) - 1
        kernel_sizes_ = ensure_seq_of_seq(kernel_sizes, num_conv)
        if strides is not None:
            strides_ = ensure_seq_of_seq(strides, num_conv)
        else:
            strides_ = [[1 for _ in range(len(kernel_sizes_[0]))] for __ in range(num_conv)]

        padding_modes_ = ensure_seq_of_seq(padding_modes, num_conv)
        paddings_ = ensure_seq_of_seq(paddings, num_conv)

        for i, (ch_in, ch_out) in enumerate(zip(num_channels[:-1], num_channels[1:])):
            new_conv = conv(
                ch_in,
                ch_out,
                kernel_sizes_[i],
                strides_[i],
                paddings_[i],
                padding_mode=padding_modes_[i],
            )
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
        skip_data_for = {}  # type: dict[int, torch.Tensor]
        conv_count = 1
        if "0" in self.skips:
            skip_dest = self.skips["0"]
            skip_data_for[skip_dest] = data
        result = data
        for this_layer, next_layer in zip(self.layers, self.layers[1:] + [None]):
            if isinstance(this_layer, torch.nn.modules.conv._ConvNd):
                if conv_count in skip_data_for:
                    result += skip_data_for[conv_count]

            result = this_layer(result)

            if isinstance(next_layer, torch.nn.modules.conv._ConvNd):
                if str(conv_count) in self.skips:
                    skip_dest = self.skips[str(conv_count)]
                    if skip_dest in skip_data_for:
                        skip_data_for[skip_dest] += result
                    else:
                        skip_data_for[skip_dest] = result

                conv_count += 1

        return result
