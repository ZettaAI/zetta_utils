from __future__ import annotations

from typing import Protocol, TypeVar

T = TypeVar("T")


class DataTransform(Protocol[T]):
    def __call__(self, __data: T) -> T:
        ...

    def prepare(self) -> None:
        ...
