from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class SampleIndexer(ABC, Generic[T]):  # pragma: no cover # abstract
    """
    Mapping between integer index id and the corresponding index for querying data.
    """

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __call__(self, idx: int) -> T:
        """Returns patch index at the given index count."""
