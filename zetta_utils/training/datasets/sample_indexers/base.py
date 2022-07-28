from abc import ABC, abstractmethod
from typing import Any


class SampleIndexer(ABC):
    @abstractmethod
    def __len__(self) -> int:
        """Size of the dataset"""

    @abstractmethod
    def __call__(self, idx: int) -> Any:
        """Returns full sample index at the given index ID."""
