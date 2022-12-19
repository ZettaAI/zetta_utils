# pylint: disable=missing-docstring # pragma: no cover
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, TypeVar

IndexT = TypeVar("IndexT")
DataT = TypeVar("DataT")


class Backend(ABC, Generic[IndexT, DataT]):  # pylint: disable=too-few-public-methods
    @abstractmethod
    def read(self, idx: IndexT) -> DataT:
        """Reads data from the given index"""

    @abstractmethod
    def write(self, idx: IndexT, data: DataT):
        """Writes given data to the given index"""

    @abstractmethod
    def clone(self, **kwargs: Dict[str, Any]) -> Backend[IndexT, DataT]:
        """Clones the Backend with the kwargs being passed to the backend."""

    name: str
