"""Basic type definitions used for type annotations."""
from __future__ import annotations

from inspect import currentframe
from types import FrameType
from typing import Any, Protocol, Sequence, Type, TypeVar, Union, overload

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


StrT = TypeVar("StrT", bound=str)
NumT = TypeVar("NumT", bound=float)
MixedT = TypeVar("MixedT", bound=Union[Sequence[float], str])


@overload
def ensure_seq_of_seq(
    x: Sequence[NumT],
    length: int,
) -> list[Sequence[NumT]]:
    ...


@overload
def ensure_seq_of_seq(
    x: Sequence[Sequence[NumT]],
    length: int,
) -> Sequence[Sequence[NumT]]:
    ...


@overload
def ensure_seq_of_seq(
    x: StrT,
    length: int,
) -> list[StrT]:
    ...


@overload
def ensure_seq_of_seq(
    x: Sequence[StrT],
    length: int,
) -> Sequence[StrT]:
    ...


@overload
def ensure_seq_of_seq(
    x: Sequence[MixedT],
    length: int,
) -> Sequence[MixedT]:
    ...


def ensure_seq_of_seq(x, length):
    """
    Replicates the argument to be a sequence (of sequences) of the given length
    if it is not already. If the argument is a sequence of sequences of a wrong length,
    `ValueError` is raised.

    Only works for sequences of basic types. Can be extended to support more types,
    but type overloades would have to be updated.
    """

    if not isinstance(x, str) and isinstance(x, Sequence) and isinstance(x[0], Sequence):
        if len(x) != length:
            raise ValueError(f"Expected sequence of {length} entries, but got {len(x)}: {x}")
        result = x
    else:
        result = [x] * length

    return result
