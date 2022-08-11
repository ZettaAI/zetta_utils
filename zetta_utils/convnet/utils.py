from typing import Optional, Sequence
import fsspec  # type: ignore
import torch


def save_model(model: torch.nn.Module, path: str):  # pragma: no cover
    torch.save(model.state_dict(), path)


def load_model(
    model: torch.nn.Module,
    ckpt_path: Optional[str] = None,
    component_names: Optional[Sequence[str]] = None,
):  # pragma: no cover
    with fsspec.open(ckpt_path) as f:
        loaded_state_raw = torch.load(f)["state_dict"]

        if component_names is None:
            loaded_state = loaded_state_raw
        else:
            loaded_state = {
                k: v
                for k, v in loaded_state_raw.items()
                if k.startswith(tuple(f"{e}." for e in component_names))
            }
        model.load_state_dict(loaded_state, strict=False)
