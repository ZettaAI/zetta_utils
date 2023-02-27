from __future__ import annotations

import attrs
import torch
from typeguard import typechecked

from zetta_utils import builder, convnet, tensor_ops


@builder.register("SimpleInferenceRunner")
@attrs.mutable
@typechecked
class SimpleInferenceRunner:  # pragma: no cover
    # Don't create the model during initialization for efficient serialization
    model_path: str
    unsqueeze_to: int | None = None

    def __call__(self, src: torch.Tensor) -> torch.Tensor:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        # load model during the call _with caching_
        model = convnet.utils.load_model(self.model_path, device=device, use_cache=True)
        if self.unsqueeze_to is not None:
            src = tensor_ops.unsqueeze_to(src, self.unsqueeze_to)
        result = model(src.to(device))
        return result
