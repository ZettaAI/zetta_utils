"""Tensor type annotations."""
from __future__ import annotations

from typing import TypeVar, Union

import numpy.typing as npt
import torch

TensorTypeVar = TypeVar(
    "TensorTypeVar", torch.Tensor, npt.NDArray, #Union[torch.Tensor, npt.NDArray]
)
Tensor = Union[torch.Tensor, npt.NDArray]
