# pylint: disable=missing-docstring # pragma: no cover
from abc import ABC, abstractmethod


class BaseDataBackend(ABC):  # pylint: disable=too-few-public-methods
    @abstractmethod
    def read(self, idx):
        """Reads data from the given index"""

    @abstractmethod
    def write(self, idx, value):
        """Writes given value to the given index"""
