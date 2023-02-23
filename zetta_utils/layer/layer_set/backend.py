# pylint: disable=missing-docstring
from __future__ import annotations

from typing import TypeVar

import attrs

from .. import Backend, Layer

IndexT = TypeVar("IndexT")
DataT = TypeVar("DataT")


@attrs.frozen
class LayerSetBackend(Backend[IndexT, dict[str, DataT]]):  # pylint: disable=too-few-public-methods
    layers: dict[str, Layer[IndexT, DataT]]

    def read(self, idx: IndexT) -> dict[str, DataT]:
        return {k: v.read_with_procs(idx) for k, v in self.layers.items()}

    def write(self, idx: IndexT, data: dict[str, DataT]):
        for k, v in data.items():
            self.layers[k].write_with_procs(idx, v)
