# pylint: disable=missing-docstring
from __future__ import annotations

from typing import Mapping, TypeVar

import attrs

from .. import Backend, Layer

IndexT = TypeVar("IndexT")
DataT = TypeVar("DataT")
DataWriteT = TypeVar("DataWriteT")


@attrs.frozen
class LayerSetBackend(
    Backend[IndexT, dict[str, DataT], dict[str, DataWriteT]]
):  # pylint: disable=too-few-public-methods
    layers: Mapping[str, Layer[IndexT, DataT, DataWriteT]]

    def read(self, idx: IndexT) -> dict[str, DataT]:
        return {k: v.read_with_procs(idx) for k, v in self.layers.items()}

    def write(self, idx: IndexT, data: dict[str, DataWriteT]):
        for k, v in data.items():
            self.layers[k].write_with_procs(idx, v)

    @property
    def name(self) -> str:
        return f"LayerSet[f{'_'.join(self.layers.keys())}]"  # pragma: no cover

    def with_changes(self, **kwargs) -> LayerSetBackend[IndexT, DataT, DataWriteT]:
        return attrs.evolve(self, **kwargs)  # pragma: no cover
