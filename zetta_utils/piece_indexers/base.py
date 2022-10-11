from abc import ABC, abstractmethod
from typing import Any


class PieceIndexer(ABC):
    @abstractmethod
    def __len__(self) -> int:
        """Total number of patches."""

    @abstractmethod
    def __call__(self, idx: int) -> Any:
        """Returns patch index at the given index count."""
