from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Iterable
from . import LayerIndex

RawIndexT = TypeVar("RawIndexT")
IndexT = TypeVar("IndexT", bound=LayerIndex)
DataT = TypeVar("DataT")


class IndexAdjuster(ABC, Generic[IndexT]):
    @abstractmethod
    def __call__(self, idx: IndexT) -> IndexT:
        """
        Modifies incoming canonical index.
        """


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

class IndexChunker(ABC, Generic[IndexT]):
    @abstractmethod
    def __call__(self, idx: IndexT) -> Iterable[IndexT]:
        """
        Gets chunks from the given index.
        """

class IdentityIndexChunker(IndexChunker[IndexT]):
    def __call__(self, idx: IndexT) -> Iterable[IndexT]:
        return [idx]
