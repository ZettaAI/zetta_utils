# pylint: disable=missing-docstring
"""Common Layer Properties."""
from __future__ import annotations


class BaseLayer:
    """Base Layer class."""

    def __init__(
        self,
        index_range_specs=None,
        readonly: bool = True,
        read_index_adjs: list = None,
        read_postprocs: list = None,
    ):
        self.readonly = readonly
        self.index_range_specs = index_range_specs

        if read_index_adjs is None:
            read_index_adjs = []
        if read_postprocs is None:
            read_postprocs = []

        self.read_index_adjs = read_index_adjs
        self.postprocessors = read_postprocs

        if index_range_specs is not None:
            raise NotImplementedError("Index range layer specs is not yet implemented.")

    def write(self, idx, value):
        if self.readonly:
            raise IOError(f"Attempting to write to a read only layer {self}")
        self._write(idx=idx, value=value)

    def read(self, idx):
        for adj in self.read_index_adjs:
            idx = adj(idx)
        res = self._read(idx=idx)
        for pproc in self.postprocessors:
            res = pproc(res)
        return res

    def _write(self, idx, value):
        raise NotImplementedError("`_write` method not overriden for a layer type.")

    def _read(self, idx):
        raise NotImplementedError("`_read` method not overriden for a layer type.")

    def __repr__(self):
        raise NotImplementedError("`__repr__` method not overriden for a layer type.")
