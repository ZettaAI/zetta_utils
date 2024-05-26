from __future__ import annotations

from typing import Iterable

from numpy import typing as npt

from zetta_utils import builder


@builder.register("apply_mask_fn")
def apply_mask_fn(
    src: npt.NDArray,
    masks: Iterable[npt.NDArray],
    fill_value: float = 0,
) -> npt.NDArray:
    result = src
    for mask in masks:
        result[mask > 0] = fill_value
    result = result.astype(src.dtype)
    return result
