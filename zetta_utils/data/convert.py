"""Type conversion functions."""
from __future__ import annotations

import torch
import numpy as np
import numpy.typing as npt
from typeguard import typechecked

import zetta_utils as zu


@typechecked
def to_np(a: zu.typing.Tensor) -> npt.NDArray:
    """
    Convert the given array to numpy.ndarray.

    Args:
        a (zu.types.Tensor): Input array.

    Returns:
        npt.NDArray: Input array in numpy.ndarray format.
    """
    if isinstance(a, torch.Tensor):
        result = a.cpu().detach().numpy()
    elif isinstance(a, np.ndarray):
        result = a
    else:
        assert False, "Type checking error"  # pragma: no cover
    return result


@typechecked
def to_torch(a: zu.typing.Tensor) -> torch.Tensor:
    """
    Convert the given array to torch tensor.

    Args:
        a (zu.types.Tensor): Input array.

    Returns:
        torch.Tensor: Input array in torch tensro format.
    """
    if isinstance(a, torch.Tensor):
        result = a
    elif isinstance(a, np.ndarray):
        result = torch.from_numpy(a)
    else:
        assert False, "Type checking error"  # pragma: no cover
    return result
