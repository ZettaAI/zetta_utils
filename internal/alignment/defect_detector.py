from __future__ import annotations

import attrs
import einops
import numpy as np
import torch
from numpy import typing as npt
from typeguard import typechecked

from zetta_utils import builder, convnet
from zetta_utils.tensor_ops import convert


@builder.register("DefectDetector")
@typechecked
@attrs.mutable
class DefectDetector:
    # Input uint8 [   0 .. 255]
    # Output uint8 Prediction [0 .. 255]

    # Don't create the model during initialization for efficient serialization
    model_path: str

    def __call__(self, src: npt.NDArray) -> npt.NDArray:
        if (src != 0).sum() == 0:
            return np.full_like(src, 0).astype(np.uint8)
        else:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

            # load model during the call _with caching_
            model = convnet.utils.load_model(self.model_path, device=device, use_cache=True)
            if src.dtype == np.uint8:
                data_in = convert.to_torch(src).float() / 255.0  # [0.0 .. 1.0]
            else:
                raise ValueError(f"Unsupported src dtype: {src.dtype}")

            data_in = einops.rearrange(data_in, "C X Y Z -> Z C X Y").to(device=device)

            with torch.no_grad():
                result = model(data_in)
            result = einops.rearrange(result, "Z C X Y -> C X Y Z")
            result = 255.0 * torch.sigmoid(result)

        return convert.to_np(result.round().clamp(0, 255).type(torch.uint8))
