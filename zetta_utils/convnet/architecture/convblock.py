# pylint: disable=protected-access
from __future__ import annotations

from typing import Callable, Literal, Sequence, Union

import torch
from torch import nn
from typeguard import typechecked
from typing_extensions import TypeAlias

from zetta_utils import builder
from zetta_utils.tensor_ops import crop_center
from zetta_utils.typing import ensure_seq_of_seq

Padding: TypeAlias = Union[Literal["same", "valid"], Sequence[int]]
PaddingMode: TypeAlias = Literal["zeros", "reflect", "replicate", "circular"]
ActivationMode: TypeAlias = Literal["pre", "post"]


def _get_size(data: torch.Tensor) -> Sequence[int]:
    # In tracing mode, shapes obtained from tensor.shape are traced as tensors
    if isinstance(data.shape[0], torch.Tensor):  # type: ignore[unreachable] # pragma: no cover
        size = list(map(lambda x: x.item(), data.shape))  # type: ignore[unreachable]
    else:
        size = data.shape
    return size


@builder.register("ConvBlock", versions=">=0.0.2")
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
        first convolution (index 1) to the output of third convolution (index 3).
        0 specifies the input to the first layer.
    :param normalize_last: Whether to apply normalization after the last layer.
    :param activate_last: Whether to apply activation after the last layer.

    :param padding_modes: Convolution padding modes. When specified as a single string,
        it will be passed as the padding mode parameter to all convolution constructors.
        When specified as a sequence of strings, each item in the sequence will be passed as
        padding mode to the corresponding convolution in order. The list length must match
        the number of convolutions.

    :param activation_mode: Whether to use ``pre-activation`` (norm. -> act. -> conv.) or
        ``post-activation`` (conv. -> norm. -> act.) for residual skip connections
        (He et al. 2016).
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
        activation_mode: ActivationMode = "post",
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
            strides_ = [[1] * len(kernel_sizes_[0])] * num_conv

        padding_modes_ = ensure_seq_of_seq(padding_modes, num_conv)
        paddings_ = ensure_seq_of_seq(paddings, num_conv)

        # Manage skip connections for pre-activation mode
        pre_skips_src = [0]
        pre_skips_dst = []

        # Manage skip connections for post-activation mode
        post_skips_src = [0]
        post_skips_dst = []

        for i, (ch_in, ch_out) in enumerate(zip(num_channels[:-1], num_channels[1:])):
            if (activation_mode == "pre") or (i >= 1):
                if normalization is not None:
                    self.layers.append(normalization(ch_in))

                post_skips_dst.append(len(self.layers))
                self.layers.append(activation())
                post_skips_src.append(len(self.layers))

            new_conv = conv(
                ch_in,
                ch_out,
                kernel_sizes_[i],
                strides_[i],
                paddings_[i],
                padding_mode=padding_modes_[i],
            )
            # TODO: make this step optional
            if new_conv.bias is not None:
                new_conv.bias.data[:] = 0
            self.layers.append(new_conv)

            pre_skips_src.append(len(self.layers))
            pre_skips_dst.append(len(self.layers))

        if normalize_last and (normalization is not None):
            self.layers.append(normalization(num_channels[-1]))
        if activation_mode == "post":
            post_skips_dst.append(len(self.layers))
        if activate_last:
            self.layers.append(activation())

        # Pre- or post-activation
        self.skips_src = pre_skips_src if activation_mode == "pre" else post_skips_src
        self.skips_dst = pre_skips_dst if activation_mode == "pre" else post_skips_dst

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        skip_data_for = {}  # type: dict[int, torch.Tensor]
        result = data
        conv_count = 0
        for i, layer in enumerate(self.layers):
            if (i in self.skips_dst) and (conv_count in skip_data_for):
                size = _get_size(result)
                result += crop_center(skip_data_for[conv_count], size)

            if (i in self.skips_src) and (str(conv_count) in self.skips):
                skip_dest = self.skips[str(conv_count)]
                if skip_dest in skip_data_for:
                    size = _get_size(result)
                    skip_data_for[skip_dest] += crop_center(result, size)
                else:
                    skip_data_for[skip_dest] = result

            result = layer(result)

            if isinstance(layer, torch.nn.modules.conv._ConvNd):
                conv_count += 1

        if (len(self.layers) in self.skips_dst) and (conv_count in skip_data_for):
            size = _get_size(result)
            result += crop_center(skip_data_for[conv_count], size)

        return result
