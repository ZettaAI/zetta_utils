# pylint: disable=protected-access
from __future__ import annotations

from typing import (
    Callable,
    Dict,
    Hashable,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import torch
from torch import nn
from typeguard import typechecked
from typing_extensions import TypeGuard

from zetta_utils import builder

Padding = Union[Literal["same", "valid"], int, Tuple[int, ...]]
PaddingMode = Literal["zeros", "reflect", "replicate", "circular"]

_T = TypeVar("_T", bound=Hashable)  # in lieu of ``Immutable`` type


def _is_list(x: _T | List[_T]) -> TypeGuard[List[_T]]:
    return isinstance(x, List)


def _is_not_list(x: _T | List[_T]) -> TypeGuard[_T]:
    # TypeGuard only ensures type for positive outcome
    return not isinstance(x, List)


@overload
def _ensure_list(x: _T, length: int) -> List[_T]:
    ...


@overload
def _ensure_list(x: List[_T], length: int) -> List[_T]:
    ...


def _ensure_list(x, length):
    if _is_list(x):
        if len(x) != length:
            raise ValueError(f"Expected list of {length} entries, but got {len(x)}: {x}")
        x_list = x
    elif _is_not_list(x):
        x_list = [x] * length
    return x_list


@builder.register("ConvBlock")
@typechecked
# cant use attrs because torch Module + attrs combination makes pylint freak out
class ConvBlock(nn.Module):
    """
    A block with sequential convolutions with activations, optional normalization and residual
    connections.

    :param num_channels: List of integers specifying the number of channels of each
        convolution. For example, specification [1, 2, 3] will correspond to a sequence of
        2 convolutions, where the first one has 1 input channel and 2 output channels and
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
    :param paddings: Convolution padding sizes. Can accept "same", "valid", a single integer,
        a tuple, or a list of any of these. When specified as a single string, integer, or a
        tuple, it will be passed as the padding parameter to all convolution constructors.
        When specified as a list, each item in the list will be passed as padding to the
        corresponding convolution in order. The list length must match the number of
        convolutions.
    :param skips: Specification for residual skip connection. For example,
        ``skips={"1": 3}`` specifies a single residual skip connection from the output of the
        first convolution (index 1) to the input of third convolution (index 3).
        0 specifies the input to the first layer.
    :param normalize_last: Whether to apply normalization after the last layer.
    :param activate_last: Whether to apply activation after the last layer.
    :param padding_modes: Convolution padding modes. Accepts "zeros" (default), "reflect",
        "replicate" and "circular". When specified as a single string, it will be passed as
        the padding mode parameter to all convolution constructors.
        When specified as a list, each item in the list will be passed as padding mode to
        the corresponding convolution in order. The list length must match the number of
        convolutions.
    """

    def __init__(
        self,
        num_channels: List[int],
        activation: Callable[[], torch.nn.Module] = torch.nn.LeakyReLU,
        conv: Callable[..., torch.nn.modules.conv._ConvNd] = torch.nn.Conv2d,
        normalization: Optional[Callable[[int], torch.nn.Module]] = None,
        kernel_sizes: Union[int, Tuple[int, ...], List[Union[int, Tuple[int, ...]]]] = 3,
        strides: Union[int, Tuple[int, ...], List[Union[int, Tuple[int, ...]]]] = 1,
        paddings: Union[Padding, List[Padding]] = "same",
        skips: Optional[Dict[str, int]] = None,
        normalize_last: bool = False,
        activate_last: bool = False,
        padding_modes: Union[PaddingMode, List[PaddingMode]] = "zeros",
    ):  # pylint: disable=too-many-locals
        super().__init__()
        if skips is None:
            self.skips = {}
        else:
            self.skips = skips
        self.layers = torch.nn.ModuleList()

        num_conv = len(num_channels) - 1
        kernel_sizes_ = _ensure_list(kernel_sizes, num_conv)
        strides_ = _ensure_list(strides, num_conv)
        paddings_ = _ensure_list(paddings, num_conv)
        padding_modes_ = _ensure_list(padding_modes, num_conv)

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
        skip_data_for = {}  # type: Dict[int, torch.Tensor]
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
