"""Basic type definitions used for type annotations."""
from __future__ import annotations

import typing
import torch
import numpy.typing as npt


Array = typing.Union[torch.Tensor, npt.NDArray]
