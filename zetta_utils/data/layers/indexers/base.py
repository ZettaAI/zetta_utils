# pylint: disable=missing-docstring # pragma: no cover
from abc import ABC, abstractmethod
from typing import Iterable, Callable, Tuple, Any, Literal


class BaseIndexer(ABC):  # pylint: disable=too-few-public-methods
    @abstractmethod
    def __call__(
        self, idx, mode: Literal["read", "write"]
    ) -> Tuple[Any, Iterable[Callable]]:
        """Returns an updated index, as well as an Iterable of Callable processors
        to be applied to the data before writing or after reading."""
