import cachetools
import fsspec
import numpy as np
import onnx
import onnxruntime as ort
import torch

from zetta_utils import builder

_session_cache: cachetools.LRUCache = cachetools.LRUCache(maxsize=1)


@cachetools.cached(_session_cache)
def _get_session(model_path: str):  # pragma: no cover
    with fsspec.open(model_path, "rb") as f:
        if model_path.endswith(".onnx"):
            onnx_model = onnx.load(f)
    return ort.InferenceSession(
        onnx_model.SerializeToString(), providers=["CUDAExecutionProvider"]
    )


@builder.register("run_affinities_inference")
def run_affinities_inference(
    image: torch.Tensor,
    image_mask: torch.Tensor,
    output_mask: torch.Tensor,
    model_path: str,
    myelin_mask_threshold: float,
) -> torch.Tensor:  # pragma: no cover

    session = _get_session(model_path)
    output = session.run(None, {"input": (image * image_mask).unsqueeze(0).float().numpy()})[0]

    aff = output[:, :3, ...]
    msk = np.amax(output[:, 3:, ...], axis=-4, keepdims=True)
    output = np.concatenate((aff, msk), axis=-4)
    output = torch.Tensor(output[0])
    output_aff = output[0:3, :, :, :]
    output_mye_mask = output[3:, :, :, :] < myelin_mask_threshold
    output = torch.permute(torch.Tensor(output_aff) * output_mask * output_mye_mask, (0, 2, 3, 1))

    return output
