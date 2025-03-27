from __future__ import annotations

import gc
from typing import ClassVar

import attrs
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
    unsqueeze_to: int | None = None
    cleanup_frequency: int = 100

    call_count: ClassVar[int] = 0

    def __call__(self, src: Tensor) -> npt.NDArray:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        # load model during the call _with caching_
        model = convnet.utils.load_model(self.model_path, device=device, use_cache=True).eval()

        if self.unsqueeze_to is not None:
            src = tensor_ops.unsqueeze_to(src, self.unsqueeze_to)
        with torch.inference_mode():
            result = to_np(model(to_torch(src, device)))

        # Prevent GPU memory leaks
        if device == "cuda":
            type(self).call_count += 1
            if type(self).call_count % self.cleanup_frequency == 0:
                gc.collect()
                torch.cuda.empty_cache()
        return result
