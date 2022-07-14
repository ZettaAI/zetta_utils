"""Definitions used for type annotations."""

from __future__ import annotations

import typing
import torch
import numpy as np
import numpy.typing as npt


Array = typing.Union[torch.Tensor, npt.NDArray]


def to_np(a: Array) -> npt.NDArray:
    """
    Convert the given array to numpy.

    Args:
        a (zu.types.Array): Input array.

    Returns:
        npt.NDArray: Input array in numpy format.
    """
    if isinstance(a, torch.Tensor):
        return a.cpu().detach().numpy()

    if isinstance(a, np.ndarray):
        return a
    # else:
    raise ValueError(f"Expected input of type {Array}, got {type(a)}")
