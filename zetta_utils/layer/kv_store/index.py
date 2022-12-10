# pylint: disable=missing-docstring
from __future__ import annotations

from abc import abstractmethod
from typing import List, Tuple, Union

from .. import IndexConverter, LayerIndex


class KVIndex(LayerIndex):
    @classmethod
    @abstractmethod
    def default_convert(cls, idx_raw: RawKVIndex) -> KVIndex:
        ...


KeyIndex = Union[str, List[str]]
KeyAndAttributeIndex = Tuple[KeyIndex, Tuple[str]]

ConvertibleKVIndex = Union[KeyIndex, KeyAndAttributeIndex]
RawKVIndex = Union[ConvertibleKVIndex, KVIndex]


class KVIndexConverter(IndexConverter[RawKVIndex, KVIndex]):
    @abstractmethod
    def __call__(self, idx_raw: RawKVIndex) -> KVIndex:
        ...
