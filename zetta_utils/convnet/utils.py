from __future__ import annotations

import io
from typing import Optional, Sequence, TypeAlias, Union

import cachetools
import fsspec
import onnx
import onnxruntime as ort
import torch
from typeguard import typechecked

from zetta_utils import builder, log, tensor_ops

logger = log.get_logger("zetta_utils")

OrtInferenceSessionT: TypeAlias = ort.capi.onnxruntime_inference_collection.InferenceSession


@builder.register("load_model")
@typechecked
def load_model(
    path: str, device: Union[str, torch.device] = "cpu", use_cache: bool = False
) -> torch.nn.Module | OrtInferenceSessionT:  # pragma: no cover
    if use_cache:
        result = _load_model_cached(path, device)
    else:
        result = _load_model(path, device)
    return result


def _load_model(
    path: str, device: Union[str, torch.device] = "cpu"
) -> torch.nn.Module | OrtInferenceSessionT:  # pragma: no cover
    logger.debug(f"Loading model from '{path}'")
    if path.endswith(".json"):
        result = builder.build(path=path).to(device)
    elif path.endswith(".jit"):
        with fsspec.open(path, "rb") as f:
            result = torch.jit.load(f, map_location=device)
    elif path.endswith(".onnx"):
        providers = ["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        with fsspec.open(path, "rb") as f:
            result = ort.InferenceSession(onnx.load(f).SerializeToString(), providers=providers)
    else:
        raise ValueError(f"Unsupported file format: {path}")

    return result


_load_model_cached = cachetools.cached(cachetools.LRUCache(maxsize=2))(_load_model)


@typechecked
def save_model(model: torch.nn.Module, path: str):  # pragma: no cover
    bytesbuffer = io.BytesIO()
    torch.save(model.state_dict(), bytesbuffer)
    with fsspec.open(path, "wb") as f:
        f.write(bytesbuffer.getvalue())


@builder.register("load_weights_file")
@typechecked
def load_weights_file(
    model: torch.nn.Module,
    ckpt_path: Optional[str] = None,
    component_names: Optional[Sequence[str]] = None,
    remove_component_prefix: bool = True,
    strict: bool = True,
) -> torch.nn.Module:  # pragma: no cover
    if ckpt_path is None:
        return model

    with fsspec.open(ckpt_path) as f:
        loaded_state_raw = torch.load(f)["state_dict"]
        if component_names is None:
            loaded_state = loaded_state_raw
        elif remove_component_prefix:
            loaded_state = {}
            for e in component_names:
                for k, x in loaded_state_raw.items():
                    if k.startswith(f"{e}."):
                        new_k = k[len(f"{e}.") :]
                        loaded_state[new_k] = x
        else:
            loaded_state = {
                k: v
                for k, v in loaded_state_raw.items()
                if k.startswith(tuple(f"{e}." for e in component_names))
            }
        model.load_state_dict(loaded_state, strict=strict)
    return model


@typechecked
def load_and_run_model(
    path: str,
    data_in: torch.Tensor,
    device: Union[str, torch.device, None] = None,
    use_cache: bool = True,
) -> torch.Tensor:  # pragma: no cover

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    autocast_device = device.type if isinstance(device, torch.device) else str(device)

    model = load_model(path=path, device=device, use_cache=use_cache)

    if path.endswith(".onnx"):
        data_in_np = data_in.numpy()
        # No idea why I'm getting `error: "Tensor" not callable  [operator]` here
        # reveal_type(data_in) = torch._tensor.Tensor
        output_np = model.run(None, {"input": data_in_np})[0]  # type: ignore
        output = tensor_ops.convert.to_torch(output_np)
    else:
        with torch.autocast(device_type=autocast_device):
            output = model(data_in.to(device))
    return output
