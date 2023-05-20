# pylint: disable=missing-docstring
"""Common Layer Properties."""
from __future__ import annotations

from typing import Any, Generic, Iterable, TypeVar, Union

import attrs

from . import Backend, DataProcessor, IndexProcessor, JointIndexDataProcessor

BackendIndexT = TypeVar("BackendIndexT")
BackendDataT = TypeVar("BackendDataT")
BackendT = TypeVar("BackendT", bound=Backend)
LayerT = TypeVar("LayerT", bound="Layer")


@attrs.frozen
class Layer(Generic[BackendIndexT, BackendDataT]):
    backend: Backend[BackendIndexT, BackendDataT]
    readonly: bool = False

    index_procs: tuple[IndexProcessor[BackendIndexT], ...] = ()
    read_procs: tuple[
        Union[DataProcessor[BackendDataT], JointIndexDataProcessor[BackendDataT, BackendIndexT]],
        ...,
    ] = ()
    write_procs: tuple[
        Union[DataProcessor[BackendDataT], JointIndexDataProcessor[BackendDataT, BackendIndexT]],
        ...,
    ] = ()

    def read_with_procs(
        self,
        idx: BackendIndexT,
    ) -> BackendDataT:
        idx_proced = idx
        for proc_idx in self.index_procs:
            idx_proced = proc_idx(idx_proced)

        for e in reversed(self.read_procs):
            if isinstance(e, JointIndexDataProcessor):
                idx_proced = e.process_index(idx=idx_proced, mode="read")

        data_backend = self.backend.read(idx=idx_proced)

        data_proced = data_backend
        for e in self.read_procs:
            if isinstance(e, JointIndexDataProcessor):
                data_proced = e.process_data(data=data_proced, mode="read")
            else:
                data_proced = e(data_proced)

        return data_proced

    def write_with_procs(
        self,
        idx: BackendIndexT,
        data: BackendDataT,
    ):
        if self.readonly:
            raise IOError(f"Attempting to write to a read only layer {self}")

        idx_proced = idx
        for proc_idx in self.index_procs:
            idx_proced = proc_idx(idx_proced)

        for e in self.write_procs:
            if isinstance(e, JointIndexDataProcessor):
                idx_proced = e.process_index(idx=idx_proced, mode="write")

        data_proced = data
        for e in self.write_procs:
            if isinstance(e, JointIndexDataProcessor):
                data_proced = e.process_data(data=data_proced, mode="write")
            else:
                data_proced = e(data_proced)

        self.backend.write(idx=idx_proced, data=data_proced)

    @property
    def name(self) -> str:  # pragma: no cover
        return self.backend.name

    def with_procs(
        self: LayerT,
        index_procs: Iterable[IndexProcessor[BackendIndexT]] | None = None,
        read_procs: Iterable[
            Union[
                DataProcessor[BackendDataT], JointIndexDataProcessor[BackendDataT, BackendIndexT]
            ]
        ]
        | None = None,
        write_procs: Iterable[
            Union[
                DataProcessor[BackendDataT], JointIndexDataProcessor[BackendDataT, BackendIndexT]
            ]
        ]
        | None = None,
    ) -> LayerT:
        proc_mods = {}  # type: dict[str, Any]
        if index_procs is not None:
            proc_mods["index_procs"] = tuple(index_procs)
        if read_procs is not None:
            proc_mods["read_procs"] = tuple(read_procs)
        if write_procs is not None:
            proc_mods["write_procs"] = tuple(write_procs)

        return attrs.evolve(self, **proc_mods)
