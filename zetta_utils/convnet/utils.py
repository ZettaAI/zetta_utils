from __future__ import annotations

import io
from typing import Literal, Optional, Sequence, Union, overload

import cachetools
import fsspec
import onnx
import onnx2torch
import torch
from numpy import typing as npt
from typeguard import typechecked

from zetta_utils import builder, log, tensor_ops

logger = log.get_logger("zetta_utils")


@builder.register("load_model")
@typechecked
def load_model(
    path: str, device: Union[str, torch.device] = "cpu", use_cache: bool = False
) -> torch.nn.Module:  # pragma: no cover
    if use_cache:
        result = _load_model_cached(path, device)
    else:
        result = _load_model(path, device)
    return result


def _load_model(
    path: str, device: Union[str, torch.device] = "cpu"
) -> torch.nn.Module:  # pragma: no cover
    logger.debug(f"Loading model from '{path}'")
    if path.endswith(".json"):
        result = builder.build(path=path).to(device)
    elif path.endswith(".jit"):
        with fsspec.open(path, "rb") as f:
            result = torch.jit.load(f, map_location=device)
    elif path.endswith(".onnx"):
        with fsspec.open(path, "rb") as f:
            result = onnx2torch.convert(onnx.load(f)).to(device)
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
        # Scheduler might not have GPU, but will still attempt to load model weights
        map_location = "cpu" if not torch.cuda.is_available() else None
        loaded_state_raw = torch.load(f, map_location=map_location)["state_dict"]
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


@overload
def load_and_run_model(
    path: str,
    data_in: torch.Tensor,
    device: Union[Literal["cpu", "cuda"], torch.device, None] = ...,
    use_cache: bool = ...,
) -> torch.Tensor:
    ...


@overload
def load_and_run_model(
    path: str,
    data_in: npt.NDArray,
    device: Union[Literal["cpu", "cuda"], torch.device, None] = ...,
    use_cache: bool = ...,
) -> npt.NDArray:
    ...


@typechecked
def load_and_run_model(path, data_in, device=None, use_cache=True):  # pragma: no cover

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(path=path, device=device, use_cache=use_cache)

    autocast_device = device.type if isinstance(device, torch.device) else str(device)
    with torch.inference_mode():  # uses less memory when used with JITs
        with torch.autocast(device_type=autocast_device):
            output = model(tensor_ops.convert.to_torch(data_in, device=device))
            output = tensor_ops.convert.astype(output, reference=data_in)
    return output
