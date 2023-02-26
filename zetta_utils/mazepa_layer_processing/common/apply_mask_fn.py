from __future__ import annotations

from typing import Iterable

import torch

from zetta_utils import builder


@builder.register("apply_mask_fn")
def apply_mask_fn(
    src: torch.Tensor,
    masks: Iterable[torch.Tensor],
    fill_value: float = 0,
) -> torch.Tensor:
    result = src
    for mask in masks:
        result[mask > 0] = fill_value
    result = result.to(src.dtype)
    return result
