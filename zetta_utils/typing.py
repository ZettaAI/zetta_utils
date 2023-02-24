"""Basic type definitions used for type annotations."""
from __future__ import annotations

from inspect import currentframe
from types import FrameType
from typing import Any, Protocol, Type, TypeVar

import typeguard


def get_orig_class(obj: Any) -> Type:  # pragma: no cover
    """
    Helper function to determine the original class for a generic subclass. This function
    is necessary because `obj.__orig_class__` does not work during `__init__`.
    Taken verbatim from https://github.com/Stewori/pytypes/blob/master/pytypes/type_util.py
    """
    try:
        return object.__getattribute__(obj, "__orig_class__")
    except AttributeError:
        cls = object.__getattribute__(obj, "__class__")
        # frame = currentframe().f_back.f_back unrolled for mypy
        frame = currentframe()
        assert isinstance(frame, FrameType)
        frame = frame.f_back
        assert isinstance(frame, FrameType)
        frame = frame.f_back
        try:
            while frame:
                try:
                    res = frame.f_locals["self"]
                    if res.__origin__ is cls:
                        return res
                except (KeyError, AttributeError):
                    frame = frame.f_back
        finally:
            del frame
        return type(None)


def check_type(obj: Any, cls: Any) -> bool:  # pragma: no cover
    """Type checking that works better for type generics"""
    result = True
    try:
        typeguard.check_type(obj, cls)
    except typeguard.TypeCheckError:
        result = False
    return result


OtherTypeT = TypeVar("OtherTypeT")


class ArithmeticOperand(Protocol[OtherTypeT]):
    def __add__(self, other: OtherTypeT) -> ArithmeticOperand[OtherTypeT]:
        ...

    def __mul__(self, other: OtherTypeT) -> ArithmeticOperand[OtherTypeT]:
        ...
