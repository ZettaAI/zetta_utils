from __future__ import annotations

import attrs
import numpy as np
import torch
from numpy import typing as npt
from typeguard import typechecked

from zetta_utils import builder, convnet, tensor_ops
from zetta_utils.tensor_ops.convert import to_np, to_torch
from zetta_utils.tensor_typing import Tensor


@builder.register("SimpleInferenceRunner")
@attrs.mutable
@typechecked
class SimpleInferenceRunner:  # pragma: no cover
    # Don't create the model during initialization for efficient serialization
    model_path: str
    apply_sigmoid: bool = False
    is_3d: bool = False
    skip_zeros: bool = True
    output_channels: int | None = None

    def __call__(self, src: Tensor) -> npt.NDArray:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        if self.is_3d:
            src = tensor_ops.unsqueeze_to(src, 5)

        if self.skip_zeros and (src != 0).sum() == 0:
            output_shape = list(src.shape)
            if self.output_channels is not None:
                output_shape[1] = self.output_channels
            return np.zeros(output_shape)

        # load model during the call _with caching_
        model = convnet.utils.load_model(self.model_path, device=device, use_cache=True).eval()
        with torch.no_grad():
            result = model(to_torch(src).to(device))
        if self.apply_sigmoid:
            result = torch.sigmoid(result)

        result_np = to_np(result)
        return result_np


