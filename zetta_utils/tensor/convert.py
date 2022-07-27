"""Type conversion functions."""
from __future__ import annotations

from typing import overload

import torch
import numpy as np
import numpy.typing as npt
from typeguard import typechecked

import zetta_utils as zu


@typechecked
def to_np(a: zu.typing.Tensor) -> npt.NDArray:
    """
    Convert the given tensor to numpy.ndarray.

    Args:
        a (zu.types.Tensor): Input tensor.

    Returns:
        npt.NDArray: Input tensor in numpy.ndarray format.
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
    Convert the given tensor to torch tensor.

    Args:
        a (zu.types.Tensor): Input tensor.

    Returns:
        torch.Tensor: Input tensor in torch tensro format.
    """
    if isinstance(a, torch.Tensor):
        result = a
    elif isinstance(a, np.ndarray):
        result = torch.from_numpy(a)
    else:
        assert False, "Type checking error"  # pragma: no cover
    return result


@overload
def astype(
    data: torch.Tensor, reference: zu.typing.Tensor
) -> torch.Tensor:  # pylint: disable=missing-docstring # pragma: no cover
    ...


@overload
def astype(
    data: npt.NDArray, reference: zu.typing.Tensor
) -> npt.NDArray:  # pylint: disable=missing-docstring # pragma: no cover
    ...


@typechecked
def astype(data: zu.typing.Tensor, reference: zu.typing.Tensor) -> zu.typing.Tensor:
    """
    Convert the given `data` tensor to np.ndarray or torch.Tensor
    depending on the type of `reference`.

    Args:
        data (zu.typing.Tensor): Input tensor.
        reference (zu.typing.Tensor): Reference type tensor.

    Returns:
        zu.typing.Tensor: Input tensor in torch tensro format.
    """
    if isinstance(reference, torch.Tensor):
        result = zu.tensor.convert.to_torch(data)  # type: zu.typing.Tensor
    elif isinstance(reference, np.ndarray):
        result = zu.tensor.convert.to_np(data)
    return result
