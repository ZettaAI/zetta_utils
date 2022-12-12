from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Iterable, Protocol, TypeVar

import attrs

RawIndexT = TypeVar("RawIndexT")
IndexT = TypeVar("IndexT")
DataT = TypeVar("DataT")


class IndexAdjuster(Protocol[IndexT]):
    def __call__(self, idx: IndexT) -> IndexT:
        ...


class DataProcessor(Protocol[DataT]):
    def __call__(self, idx: DataT) -> DataT:
        ...


class DataWithIndexProcessor(ABC, Generic[DataT, IndexT]):
    @abstractmethod
    def __call__(
        self,
        data: DataT,
        idx: IndexT,
        idx_proced: IndexT,
    ) -> DataT:
        """
        Modifies data given both original and adjusted indexes
        """


class IndexChunker(Protocol[IndexT]):
    def __call__(self, idx: IndexT) -> Iterable[IndexT]:
        ...


@attrs.frozen
class IdentityIndexChunker(Generic[IndexT]):
    def __call__(self, idx: IndexT) -> Iterable[IndexT]:  # pragma: no cover # identity
        return [idx]
