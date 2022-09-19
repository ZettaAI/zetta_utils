# pylint: disable=protected-access
from typing import Callable, Union, Literal, Optional
from collections import defaultdict
from typeguard import typechecked
import torch
from torch import nn

from zetta_utils import builder


@builder.register("ConvBlock")
@typechecked
# cant use attrs because torch Module + attrs combination makes pylint freak out
class ConvBlock(nn.Module):
    def __init__(
        self,
        num_channels: list[int],
        activation: Callable[[], torch.nn.Module] = torch.nn.LeakyReLU,
        conv: Callable[..., torch.nn.modules.conv._ConvNd] = torch.nn.Conv2d,
        padding_mode: Union[Literal["valid"], Literal["same"]] = "same",
        normalization: Optional[Callable[[int], torch.nn.Module]] = None,
        kernel_sizes: Union[list[int], int, list[tuple[int, ...]], tuple[int, ...]] = 3,
        skips: Optional[dict[Union[int, str], int]] = None,
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
            kernel_sizes_ = kernel_sizes  # type: Union[list[int], list[tuple[int, ...]]]
        else:
            kernel_sizes_ = [kernel_sizes for _ in range(len(num_channels) - 1)]  # type: ignore

        assert len(kernel_sizes_) == (len(num_channels) - 1)

        for i, (ch_in, ch_out) in enumerate(zip(num_channels[:-1], num_channels[1:])):
            self.layers.append(conv(ch_in, ch_out, kernel_sizes_[i], padding=padding_mode))

            is_last_conv = i == len(num_channels) - 2

            if (normalization is not None) and ((not is_last_conv) or normalize_last):
                self.layers.append(normalization(ch_out))  # pylint: disable=not-callable

            if (not is_last_conv) or activate_last:
                self.layers.append(activation())

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        skip_data_for = defaultdict(
            lambda: torch.zeros_like(data, device=data.device)
        )  # type: dict[int, torch.Tensor]
        conv_count = 0

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
