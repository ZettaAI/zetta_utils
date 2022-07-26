# pylint: disable=missing-docstring
"""Common Layer Properties."""
from __future__ import annotations

from typing import Dict, Iterable, Callable, Optional, Any
import attrs
from typeguard import typechecked

import zetta_utils as zu

@typechecked
@attrs.mutable
class Layer:
    data_backend: zu.data.layers.data_backends.BaseDataBackend
    readonly: bool = False
    indexer: Optional[zu.data.layers.indexers.BaseIndexer] = None
    read_postprocs: Iterable[Callable] = attrs.Factory(list)

    def write(self, idx, value, **kwargs):
        if self.readonly:
            raise IOError(f"Attempting to write to a read only layer {self}")

        if self.indexer is not None:
            idx_final, processors = self.indexer(idx, mode='write')
        else:
            idx_final = idx
            processors = []

        value_final = value
        for proc in processors:
            value_final = proc(value_final)

        self.data_backend.write(idx=idx_final, value=value_final, **kwargs)

    def read(self, idx, **kwargs):
        if self.indexer is not None:
            idx_final, processors = self.indexer.read(idx, mode='read')
        else:
            idx_final = idx
            processors = []

        result_raw = self.data_backend.read(idx=idx_final, **kwargs)

        result = result_raw
        for proc in processors:
            result = proc(result)
        for pproc in self.read_postprocs:
            result = pproc(result)
        return result
