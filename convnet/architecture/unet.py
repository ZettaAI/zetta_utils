# pylint: disable=protected-access
from __future__ import annotations

from functools import partial
from typing import Callable, Literal, Sequence

import torch
from torch import nn
from typeguard import typechecked
from typing_extensions import TypeAlias

from zetta_utils import builder
from zetta_utils.tensor_ops import crop_center

from .convblock import ActivationMode, ConvBlock, Padding, PaddingMode

SkipConnectionMode: TypeAlias = Literal["sum", "concat"]


@builder.register("UNet", versions=">=0.0.2")
@typechecked
class UNet(nn.Module):
    """
    A basic UNet that uses ConvBlocks. Note that the downsamples are applied after the regular
    convolutions, while the upsamples are applied before the regular convolutions.

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
    :param kernel_sizes: List of convolution kernel sizes.
        Will be passed directly to ``zetta_utils.convnet.architecture.ConvBlock`` constructors.
    :param strides: List of convolution strides.
        Will be passed directly to ``zetta_utils.convnet.architecture.ConvBlock`` constructors.
    :param paddings: List of convolution padding sizes.
        Will be passed directly to ``zetta_utils.convnet.architecture.ConvBlock`` constructors.
    :param skips: Specifications for residual skip connections as documented in ConvBlock.
        Will be passed directly to ``zetta_utils.convnet.architecture.ConvBlock`` constructors.
    :param normalize_last: Whether to apply normalization after the last layer. Note that
        normalization is included by default within and at the end of the convblocks when the
        normalization layer is specified, and cannot be turned off.
    :param activate_last: Whether to apply activation after the last layer. Note that activation
        is included by default within and at the end of the convblocks and cannot be turned off.
    :param padding_modes:
        Will be passed directly to ``zetta_utils.convnet.architecture.ConvBlock`` constructors.
    :param unet_skip_mode:
        Whether skip connections are realized as residual connections (`sum`, default) or
        concatenated skip connections (`concat`).
    """

    def __init__(
        self,
        list_num_channels: Sequence[Sequence[int]],
        kernel_sizes: Sequence[int] | Sequence[Sequence[int]],
        conv: Callable[..., torch.nn.modules.conv._ConvNd] = torch.nn.Conv2d,
        strides: Sequence[int] | Sequence[Sequence[int]] | None = None,
        downsample: Callable[..., torch.nn.Module] = partial(torch.nn.AvgPool2d, kernel_size=2),
        upsample: Callable[..., torch.nn.Module] = partial(torch.nn.Upsample, scale_factor=2),
        activation: Callable[[], torch.nn.Module] = torch.nn.LeakyReLU,
        normalization: Callable[[int], torch.nn.Module] | None = None,
        paddings: Padding | Sequence[Padding] = "same",
        skips: dict[str, int] | None = None,
        normalize_last: bool = False,
        activate_last: bool = False,
        padding_modes: PaddingMode | Sequence[PaddingMode] = "zeros",
        unet_skip_mode: SkipConnectionMode = "sum",
        activation_mode: ActivationMode = "post",
    ):  # pylint: disable=too-many-locals, too-many-statements, too-many-branches
        super().__init__()
        assert len(list_num_channels) % 2 == 1
        assert downsample is not None
        assert upsample is not None

        self.layers = torch.nn.ModuleList()
        self.unet_skip_mode = unet_skip_mode

        normalize_last_ = [activation_mode == "post"] * len(list_num_channels)
        normalize_last_[-1] = normalize_last
        activate_last_ = [activation_mode == "post"] * len(list_num_channels)
        activate_last_[-1] = activate_last

        skips_in = []
        skips_out = []

        for i, num_channels in enumerate(list_num_channels):
            if i >= len(list_num_channels) // 2 + 1:
                try:
                    self.layers.append(
                        upsample(
                            in_channels=list_num_channels[i - 1][-1],
                            out_channels=list_num_channels[i][0],
                        )
                    )
                except TypeError:
                    self.layers.append(upsample())

                if (activation_mode == "post") and (normalization is not None):
                    self.layers.append(normalization(list_num_channels[i][0]))

                skips_out.append(len(self.layers))

                if activation_mode == "post":
                    self.layers.append(activation())

                # TODO: Assumes skip connection input has the same number of channels
                if self.unet_skip_mode == "concat":
                    num_channels = [2 * num_channels[0], *(num_channels[1:])]

            self.layers.append(
                ConvBlock(
                    num_channels=num_channels,
                    activation=activation,
                    conv=conv,
                    normalization=normalization,
                    kernel_sizes=kernel_sizes,
                    strides=strides,
                    paddings=paddings,
                    skips=skips,
                    normalize_last=normalize_last_[i],
                    activate_last=activate_last_[i],
                    padding_modes=padding_modes,
                    activation_mode=activation_mode,
                )
            )

            if i < len(list_num_channels) // 2:
                skips_in.append(len(self.layers) - 1)
                try:
                    self.layers.append(
                        downsample(
                            in_channels=list_num_channels[i][-1],
                            out_channels=list_num_channels[i + 1][0],
                        )
                    )
                except TypeError:
                    self.layers.append(downsample())

        self.skips = {}
        for skip_in, skip_out in zip(skips_in, skips_out[-1::-1]):
            self.skips[skip_in] = skip_out

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        skip_data_for = {}  # type: dict[int, torch.Tensor]
        result = data

        for i, layer in enumerate(self.layers):
            if i in skip_data_for:
                # In tracing mode, shapes obtained from tensor.shape are traced as tensors
                if isinstance(result.shape[0], torch.Tensor):  # type: ignore # pragma: no cover
                    size = list(map(lambda x: x.item(), result.shape))  # type: ignore
                else:
                    size = result.shape
                skip_data = crop_center(skip_data_for[i], size)
                if self.unet_skip_mode == "sum":
                    result = result + skip_data
                elif self.unet_skip_mode == "concat":
                    result = torch.hstack((skip_data, result))
            result = layer(result)

            if i in self.skips:
                skip_data_for[self.skips[i]] = result

        return result
