from __future__ import annotations

from typing import Sequence

import attrs
import einops
import numpy as np
import torch
from typeguard import typechecked

from zetta_utils import builder, convnet, tensor_ops
from zetta_utils.tensor_typing import TensorTypeVar


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

    apply_sigmoid: bool = False
    model_uses_czyx: bool = True  # To preserve compatibility with existing models

    def __call__(
        self,
        image: TensorTypeVar,
        image_mask: TensorTypeVar | None = None,
        output_mask: TensorTypeVar | None = None,
    ) -> TensorTypeVar:

        image_torch = tensor_ops.convert.to_torch(image)
        image_mask_torch = (
            tensor_ops.convert.to_torch(image_mask)
            if image_mask is not None
            else torch.ones_like(image_torch, dtype=torch.float)
        )
        output_mask_torch = (
            tensor_ops.convert.to_torch(output_mask)
            if output_mask is not None
            else torch.ones_like(image_torch, dtype=torch.float)
        )

        if image_torch.dtype == torch.uint8:
            data_in = image_torch.float() / 255.0  # [0.0 .. 1.0]
        else:
            raise ValueError(f"Unsupported image dtype: {image_torch.dtype}")

        # mask input
        data_in = data_in * image_mask_torch
        if self.model_uses_czyx:
            data_in = einops.rearrange(data_in, "C X Y Z -> C Z Y X")
        data_in = data_in.unsqueeze(0).float()

        data_out = convnet.utils.load_and_run_model(path=self.model_path, data_in=data_in)

        if self.apply_sigmoid:
            data_out = torch.sigmoid(data_out)

        assert len(data_out.shape) == 5
        assert data_out.shape[0] == 1

        # Extract requested channels
        arrays = []
        for channel in self.output_channels:
            arrays.append(data_out[:, channel, ...])
        if self.bg_mask_channel is not None:
            arrays.append(data_out[:, self.bg_mask_channel, ...])
        data_out = torch.from_numpy(np.stack(arrays, axis=1)[0])

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
        if self.model_uses_czyx:
            output = einops.rearrange(output, "C Z Y X -> C X Y Z")
        output = output * output_mask_torch

        return tensor_ops.convert.astype(output, image)
