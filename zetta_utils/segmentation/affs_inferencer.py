from __future__ import annotations

from typing import Sequence

import attrs
import einops
import numpy as np
import torch
from typeguard import typechecked

from zetta_utils import builder, convnet


@builder.register("AffinitiesInferencer")
@typechecked
@attrs.frozen
class AffinitiesInferencer:
    # Input uint8  [   0 .. 255]
    # Output float [   0 .. 255]

    # Don't create the model during initialization for efficient serialization
    model_path: str
    output_channels: Sequence[int]

    bg_mask_channel: int | None = None
    bg_mask_threshold: float = 0.0
    bg_mask_invert_threshold: bool = False

    def __call__(
        self,
        image: torch.Tensor,
        image_mask: torch.Tensor,
        output_mask: torch.Tensor,
    ) -> torch.Tensor:

        if image.dtype == torch.uint8:
            data_in = image.float() / 255.0  # [0.0 .. 1.0]
        else:
            raise ValueError(f"Unsupported image dtype: {image.dtype}")

        # mask input
        data_in = data_in * image_mask
        data_in = einops.rearrange(data_in, "C X Y Z -> C Z Y X")
        data_in = data_in.unsqueeze(0).float()

        data_out = convnet.utils.load_and_run_model(path=self.model_path, data_in=data_in)

        # Extract requested channels
        arrays = []
        for channel in self.output_channels:
            arrays.append(data_out[:, channel, ...])
        if self.bg_mask_channel is not None:
            arrays.append(data_out[:, self.bg_mask_channel, ...])
        data_out = torch.Tensor(np.stack(arrays, axis=1)[0])

        # mask output with bg_mask
        num_channels = len(self.output_channels)
        output = data_out[0:num_channels, :, :, :]
        if self.bg_mask_channel is not None:
            if self.bg_mask_invert_threshold:
                bg_mask = data_out[num_channels:, :, :, :] > self.bg_mask_threshold
            else:
                bg_mask = data_out[num_channels:, :, :, :] < self.bg_mask_threshold
            output = torch.Tensor(output) * bg_mask

        # mask output
        output = einops.rearrange(output, "C Z Y X -> C X Y Z")
        output = output * output_mask

        return output
