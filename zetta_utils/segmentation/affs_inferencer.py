import attrs
import einops
import torch
from typeguard import typechecked

from zetta_utils import builder, convnet

import cachetools
import fsspec
import numpy as np
import onnx
import onnxruntime as ort

_session_cache: cachetools.LRUCache = cachetools.LRUCache(maxsize=1)

@cachetools.cached(_session_cache)
def _get_session(model_path: str):  # pragma: no cover
    with fsspec.open(model_path, "rb") as f:
        if model_path.endswith(".onnx"):
            onnx_model = onnx.load(f)
    return ort.InferenceSession(
        onnx_model.SerializeToString(), providers=["CUDAExecutionProvider"]
    )

@builder.register("AffinitiesInferencer")
@typechecked
@attrs.mutable
class AffinitiesInferencer:
    # Input uint8 [   0 .. 255]
    # Output uint8 [   0 .. 255]

    # Don't create the model during initialization for efficient serialization
    model_path: str
    myelin_mask_threshold: float

    def __call__(self,
                 image: torch.Tensor,
                 image_mask: torch.Tensor,
                 output_mask: torch.Tensor,
    ) -> torch.Tensor:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        if image.dtype == torch.uint8:
            data_in = image.float() / 255.0  # [0.0 .. 1.0]
        else:
            raise ValueError(f"Unsupported image dtype: {image.dtype}")

        data_in = data_in * image_mask
        data_in = einops.rearrange(data_in, "C X Y Z -> C Z Y X")

        if self.model_path.endswith(".onnx"):
            model = _get_session(self.model_path)
            output = model.run(None, {"input": data_in.unsqueeze(0).float().numpy()})[0]
        else:
            # load model during the call _with caching_
            model = convnet.utils.load_model(
                self.model_path, device=device, use_cache=True
            )
            with torch.autocast(device_type=device):
                output = model(data_in.to(device))

        aff = output[:, :3, ...]
        msk = np.amax(output[:, 3:, ...], axis=-4, keepdims=True)
        output = np.concatenate((aff, msk), axis=-4)
        output = torch.Tensor(output[0])
        output_aff = output[0:3, :, :, :]
        output_mye_mask = output[3:, :, :, :] < self.myelin_mask_threshold
        output = torch.Tensor(output_aff) * output_mye_mask

        output = einops.rearrange(output, "C Z Y X -> C X Y Z")
        output = output * output_mask

        return (output*255).type(torch.uint8)
