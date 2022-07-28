# pylint: disable=missing-docstring
"""Common Layer Properties."""
from __future__ import annotations

from typing import Sequence, Callable, Optional, Union, Literal, List, Tuple

import attrs
from typeguard import typechecked

import zetta_utils as zu
from zetta_utils.io.indexes import (
    Index,
    IndexAdjusterWithProcessors,
)


@zu.spec_parser.register("Layer")
@typechecked
@attrs.mutable
class Layer:
    io_backend: zu.io.backends.IOBackend
    readonly: bool = False
    index_adjs: Sequence[Union[Callable, IndexAdjusterWithProcessors]] = attrs.Factory(list)
    index_converter: Optional[Callable] = None
    read_postprocs: Sequence[Callable] = attrs.Factory(list)
    write_preprocs: Sequence[Callable] = attrs.Factory(list)

    def _convert_index(self, idx_raw) -> Index:
        if self.index_converter is not None:
            result = self.index_converter(idx_raw)  # pylint: disable=not-callable
        else:
            result = self.io_backend.get_index_type().convert(idx_raw)
        return result

    def _apply_index_adjs(
        self, idx, mode: Literal["read", "write"]
    ) -> Tuple[Index, List[Callable]]:
        initial_procs = []  # type: list[Callable]
        for adj in self.index_adjs:  # pylint: disable=not-an-iterable
            if isinstance(adj, IndexAdjusterWithProcessors):
                idx, procs = adj(idx, mode=mode)
                initial_procs += procs
            else:
                idx = adj(idx)

        return idx, initial_procs

    def read(self, idx_raw):
        idx = self._convert_index(idx_raw)
        idx_final, initial_procs = self._apply_index_adjs(idx, "read")
        result_raw = self.io_backend.read(idx=idx_final)

        result = result_raw
        for proc in list(initial_procs) + list(self.read_postprocs):
            # import pdb; pdb.set_trace()
            result = proc(result)

        return result

    def write(self, idx_raw, value_raw):
        if self.readonly:
            raise IOError(f"Attempting to write to a read only layer {self}")

        idx = self._convert_index(idx_raw)
        idx_final, initial_procs = self._apply_index_adjs(idx, "write")

        value = value_raw
        for proc in list(initial_procs) + list(self.write_preprocs):
            value = proc(value)

        self.io_backend.write(idx=idx_final, value=value)

    def __getitem__(self, idx_raw):  # pragma: no cover
        return self.read(idx_raw)

    def __setitem__(self, idx_raw, value_raw):  # pragma: no cover
        return self.write(idx_raw, value_raw)
