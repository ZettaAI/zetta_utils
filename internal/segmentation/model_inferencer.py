from __future__ import annotations

import attrs
import numpy as np
import torch
from typeguard import typechecked

from zetta_utils import builder, convnet
from zetta_utils.tensor_ops import common, convert
from zetta_utils.tensor_typing import TensorTypeVar


@builder.register("ModelInferencer")
@typechecked
@attrs.frozen
class ModelInferencer:
    # Input uint8  [   0 .. 255]
    # Output float [   0 .. 1]

    model_path: str
    model_uses_czyx: bool = False
    apply_sigmoid: bool = False

    def __call__(
        self,
        image: TensorTypeVar,
    ) -> TensorTypeVar:
        if not image.any():
            return convert.to_float32(image)

        if len(image.shape) == 4:
            image = common.rearrange(image, "C X Y Z -> 1 C X Y Z")

        assert len(image.shape) == 5

        if self.model_uses_czyx:
            image = common.rearrange(image, "B C X Y Z -> B C Z Y X")

        if image.dtype in (torch.uint8, np.uint8):
            data_in = convert.to_float32(image) / 255.0  # [0.0 .. 1.0]
        else:
            raise ValueError(f"Unsupported image dtype: {image.dtype}")

        data_out = convnet.utils.load_and_run_model(path=self.model_path, data_in=data_in)

        if self.apply_sigmoid:
            # Converting to torch (and back) seems to be 10x faster than np-based sigmoid
            data_out = torch.sigmoid(convert.to_torch(data_out))  # type: ignore

        if self.model_uses_czyx:
            data_out = common.rearrange(data_out, "B C Z Y X -> B C X Y Z")

        # remove the batch channel if we added it
        assert len(data_out.shape) == 5
        if data_out.shape[0] == 1:
            data_out = common.rearrange(data_out, "1 C X Y Z -> C X Y Z")

        return convert.astype(data_out, image)
