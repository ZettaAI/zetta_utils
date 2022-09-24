"""Basic type definitions used for type annotations."""
from __future__ import annotations

from typing import Union, TypeVar
import torch
import numpy.typing as npt

TensorTypeVar = TypeVar(
    "TensorTypeVar", torch.Tensor, npt.NDArray, Union[torch.Tensor, npt.NDArray]
)
Tensor = Union[torch.Tensor, npt.NDArray]
