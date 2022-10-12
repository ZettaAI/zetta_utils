# pylint: disable=missing-docstring # pragma: no cover
from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from typing import Iterable, Callable, Tuple, Literal, Generic, TypeVar


class LayerIndex(metaclass=ABCMeta):  # pylint: disable=too-few-public-methods
    @classmethod
    @abstractmethod
    def default_convert(cls, idx_raw) -> LayerIndex:
        """
        Converts user given index to the given index format.
        """


RawIndexT = TypeVar("RawIndexT")
IndexT = TypeVar("IndexT", bound=LayerIndex)

# Converters might need to carry state, and implementing that with partials will not allow
# type checking. That's why we use a dedicated class. IndexConverters are pretty standard,
# and users won't have to create custom ones in everyday lives, so it was deemed worth it
# for additional type checking.
# Improvements wellcome. cc: https://github.com/python/mypy/issues/1484
class IndexConverter(ABC, Generic[RawIndexT, IndexT]):  # pylint: disable=too-few-public-methods
    @abstractmethod
    def __call__(self, idx_raw: RawIndexT) -> IndexT:
        """
        Returns an index in a canonical form expected by the backend.
        """


class IndexAdjuster(ABC, Generic[IndexT]):  # pylint: disable=too-few-public-methods
    @abstractmethod
    def __call__(self, idx: IndexT) -> IndexT:
        """
        Modifies incoming canonical index.
        """


class IndexAdjusterWithProcessors(ABC, Generic[IndexT]):  # pylint: disable=too-few-public-methods
    @abstractmethod
    def __call__(
        self, idx: IndexT, mode: Literal["read", "write"]
    ) -> Tuple[IndexT, Iterable[Callable]]:
        """
        Modifies incoming canonical index and returns it alongside with a list
        of processors to be applied to data after reading/before writing.
        """
