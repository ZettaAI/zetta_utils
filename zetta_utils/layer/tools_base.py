from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Iterable, Literal, Protocol, TypeVar

RawIndexT = TypeVar("RawIndexT")
IndexT = TypeVar("IndexT")
DataT = TypeVar("DataT")
T = TypeVar("T")


class Processor(Protocol[T]):
    def __call__(self, inputval: T) -> T:
        ...


class JointIndexDataProcessor(ABC, Generic[DataT, IndexT]):
    @abstractmethod
    def process_index(
        self,
        idx: IndexT,
        mode: Literal["read", "write"],
    ) -> IndexT:
        ...

    @abstractmethod
    def process_data(
        self,
        data: DataT,
        mode: Literal["read", "write"],
    ) -> DataT:
        ...


class IndexChunker(Protocol[IndexT]):
    def __call__(self, idx: IndexT) -> Iterable[IndexT]:
        ...
