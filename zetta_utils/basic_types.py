"""Basic type definitions used for type annotations."""
from __future__ import annotations

from typing import Union, List
import torch
import numpy.typing as npt


Array = Union[torch.Tensor, npt.NDArray]
Resolution = Union[int, List[int]]  # xy resolution, with z resolution set to default
