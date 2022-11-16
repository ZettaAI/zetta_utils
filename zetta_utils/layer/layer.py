# pylint: disable=missing-docstring
"""Common Layer Properties."""
from __future__ import annotations

from typing import Any, Callable, Generic, Iterable, List, Optional, TypeVar

import attrs

from zetta_utils import builder

from . import DataWithIndexProcessor, IndexConverter, LayerBackend, LayerIndex

IndexT = TypeVar("IndexT", bound=LayerIndex)
DataT = TypeVar("DataT")
RawIndexT = TypeVar("RawIndexT")
IndexConverterT = TypeVar("IndexConverterT", bound=IndexConverter)
T = TypeVar("T")


def _apply_procs(
    data: T,
    idx: IndexT,
    idx_proced: IndexT,
    procs: Iterable[Callable],
) -> T:
    result = data
    for proc in procs:
        if isinstance(proc, DataWithIndexProcessor):
            result = proc(data=result, idx=idx, idx_proced=idx_proced)
        else:
            result = proc(data=result)
    return result


# TODO: Generic parametrization for Read/Write data type
@builder.register("Layer")
@attrs.mutable
class Layer(Generic[RawIndexT, IndexT, DataT]):
    backend: LayerBackend[IndexT, DataT]
    readonly: bool = False
    index_converter: Optional[IndexConverter[RawIndexT, IndexT]] = None
    index_adjs: List[Callable[[IndexT], IndexT]] = attrs.field(factory=list)
    read_postprocs: List[Callable] = attrs.field(factory=list)
    write_preprocs: List[Callable] = attrs.field(factory=list)

    def _convert_index(self, idx_raw: RawIndexT) -> IndexT:
        if self.index_converter is not None:
            # Open problem: pylint doesn't see that IndexConverter is callable
            # Fixes welocme
            result = self.index_converter(idx_raw=idx_raw)  # pylint: disable=not-callable
        else:
            result = self.backend.get_index_type().default_convert(idx_raw)
        return result

    def read(self, idx_raw: RawIndexT) -> DataT:
        idx = self._convert_index(idx_raw)
        idx_proced = idx
        for adj in self.index_adjs:
            idx_proced = adj(idx_proced)

        result_raw = self.backend.read(idx=idx_proced)
        result = _apply_procs(
            data=result_raw,
            idx=idx,
            idx_proced=idx_proced,
            procs=self.read_postprocs,
        )

        return result

    # TODO: Parametrize by RawDataT type var
    def write(self, idx_raw: RawIndexT, value_raw: Any):
        if self.readonly:
            raise IOError(f"Attempting to write to a read only layer {self}")

        idx = self._convert_index(idx_raw)
        idx_proced = idx
        for adj in self.index_adjs:
            idx_proced = adj(idx_proced)

        value = _apply_procs(
            data=value_raw,
            idx=idx,
            idx_proced=idx_proced,
            procs=self.write_preprocs,
        )

        self.backend.write(idx=idx_proced, value=value)

    def __getitem__(self, idx_raw: RawIndexT) -> DataT:  # pragma: no cover
        return self.read(idx_raw)

    def __setitem__(self, idx_raw: RawIndexT, value_raw: Any):  # pragma: no cover
        return self.write(idx_raw, value_raw)
