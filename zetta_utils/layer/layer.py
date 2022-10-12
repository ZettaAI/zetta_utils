# pylint: disable=missing-docstring
"""Common Layer Properties."""
from __future__ import annotations

from typing import (
    Sequence,
    Callable,
    Optional,
    Union,
    Literal,
    List,
    Tuple,
    TypeVar,
    Generic,
    Any,
)

import attrs
from typeguard import typechecked

from zetta_utils import builder
from . import (
    LayerBackend,
    LayerIndex,
    IndexConverter,
    IndexAdjusterWithProcessors,
)

IndexT = TypeVar("IndexT", bound=LayerIndex)
RawIndexT = TypeVar("RawIndexT")
IndexConverterT = TypeVar("IndexConverterT", bound=IndexConverter)


# TODO: Generic parametrization for Read/Write data type
@builder.register("Layer")
@typechecked
@attrs.mutable
class Layer(Generic[RawIndexT, IndexT]):
    io_backend: LayerBackend[IndexT]
    readonly: bool = False
    index_converter: Optional[IndexConverter[RawIndexT, IndexT]] = None
    index_adjs: Sequence[Union[Callable, IndexAdjusterWithProcessors]] = attrs.field(factory=list)
    read_postprocs: Sequence[Callable] = attrs.field(factory=list)
    write_preprocs: Sequence[Callable] = attrs.field(factory=list)

    def _convert_index(self, idx_raw: RawIndexT) -> IndexT:
        if self.index_converter is not None:
            # Open problem: pylint doesn't see that IndexConverter is callable
            # Fixes welocme
            result = self.index_converter(idx_raw=idx_raw)  # pylint: disable=not-callable
        else:
            result = self.io_backend.get_index_type().default_convert(idx_raw)
        return result

    def _apply_index_adjs(
        self, idx, mode: Literal["read", "write"]
    ) -> Tuple[IndexT, List[Callable]]:
        initial_procs = []  # type: list[Callable]
        for adj in self.index_adjs:
            if isinstance(adj, IndexAdjusterWithProcessors):
                idx, procs = adj(idx=idx, mode=mode)
                initial_procs += procs
            else:
                idx = adj(idx=idx)
        return idx, initial_procs

    def read(self, idx_raw: RawIndexT) -> Any:
        idx = self._convert_index(idx_raw)
        idx_final, initial_procs = self._apply_index_adjs(idx=idx, mode="read")
        result_raw = self.io_backend.read(idx=idx_final)

        result = result_raw
        for proc in list(initial_procs) + list(self.read_postprocs):
            result = proc(data=result)

        return result

    def write(self, idx_raw: RawIndexT, value_raw: Any):
        if self.readonly:
            raise IOError(f"Attempting to write to a read only layer {self}")

        idx = self._convert_index(idx_raw)
        idx_final, initial_procs = self._apply_index_adjs(idx=idx, mode="write")

        value = value_raw
        for proc in list(initial_procs) + list(self.write_preprocs):
            value = proc(data=value)

        self.io_backend.write(idx=idx_final, value=value)

    def __getitem__(self, idx_raw: RawIndexT) -> Any:  # pragma: no cover
        return self.read(idx_raw)

    def __setitem__(self, idx_raw: RawIndexT, value_raw: Any):  # pragma: no cover
        return self.write(idx_raw, value_raw)
