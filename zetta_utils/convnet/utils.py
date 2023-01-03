import io
from typing import Optional, Sequence, Union

import cachetools
import fsspec
import torch
from typeguard import typechecked

from zetta_utils import builder


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
    if path.endswith(".json"):
        result = builder.build(path=path).to(device)
    elif path.endswith(".jit"):
        with fsspec.open(path, "rb") as f:
            result = torch.jit.load(f, map_location=device)

    return result


_load_model_cached = cachetools.cached(cachetools.LRUCache(maxsize=8))(_load_model)


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
