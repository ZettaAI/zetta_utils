"""Type conversion functions."""
from __future__ import annotations

import torch
import numpy as np
import numpy.typing as npt

import zetta_utils as zu


def to_np(a: zu.basic_types.Array) -> npt.NDArray:
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
    raise ValueError(f"Expected input of type {zu.basic_types.Array}, got {type(a)}")
