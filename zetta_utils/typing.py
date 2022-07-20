"""Basic type definitions used for type annotations."""
from __future__ import annotations

from typing import Union, Any
import torch
import numpy.typing as npt
import typeguard


Array = Union[torch.Tensor, npt.NDArray]


def check_type(obj: Any, cls: Any) -> bool:
    """Type checking that works for type generics"""
    result = True
    try:
        typeguard.check_type("value", obj, cls)
    except TypeError:
        result = False
    return result
