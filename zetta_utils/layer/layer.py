# pylint: disable=missing-docstring
"""Common Layer Properties."""
from __future__ import annotations

from copy import copy
from typing import Callable, Generic, Iterable, List, TypeVar, Union, overload

import attrs

from zetta_utils import builder

from . import Backend, DataProcessor, DataWithIndexProcessor, Frontend, IndexAdjuster

BackendIndexT = TypeVar("BackendIndexT")
BackendDataT = TypeVar("BackendDataT")
BackendT = TypeVar("BackendT", bound=Backend)

UserReadDataT0_co = TypeVar("UserReadDataT0_co", covariant=True)
UserReadDataT1_co = TypeVar("UserReadDataT1_co", covariant=True)
UserReadDataT2_co = TypeVar("UserReadDataT2_co", covariant=True)
UserReadDataT3_co = TypeVar("UserReadDataT3_co", covariant=True)

UserWriteDataT0_contra = TypeVar("UserWriteDataT0_contra", contravariant=True)
UserWriteDataT1_contra = TypeVar("UserWriteDataT1_contra", contravariant=True)
UserWriteDataT2_contra = TypeVar("UserWriteDataT2_contra", contravariant=True)
UserWriteDataT3_contra = TypeVar("UserWriteDataT3_contra", contravariant=True)

UserReadIndexT0_contra = TypeVar("UserReadIndexT0_contra", contravariant=True)
UserReadIndexT1_contra = TypeVar("UserReadIndexT1_contra", contravariant=True)
UserReadIndexT2_contra = TypeVar("UserReadIndexT2_contra", contravariant=True)
UserReadIndexT3_contra = TypeVar("UserReadIndexT3_contra", contravariant=True)

UserWriteIndexT0_contra = TypeVar("UserWriteIndexT0_contra", contravariant=True)
UserWriteIndexT1_contra = TypeVar("UserWriteIndexT1_contra", contravariant=True)
UserWriteIndexT2_contra = TypeVar("UserWriteIndexT2_contra", contravariant=True)
UserWriteIndexT3_contra = TypeVar("UserWriteIndexT3_contra", contravariant=True)


def _apply_procs(
    data: BackendDataT,
    idx: BackendIndexT,
    idx_proced: BackendIndexT,
    procs: Iterable[Callable],
) -> BackendDataT:
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
class Layer(
    Generic[
        BackendT,
        BackendIndexT,
        BackendDataT,
        UserReadIndexT0_contra,
        UserReadDataT0_co,
        UserWriteIndexT0_contra,
        UserWriteDataT0_contra,
        UserReadIndexT1_contra,
        UserReadDataT1_co,
        UserWriteIndexT1_contra,
        UserWriteDataT1_contra,
        UserReadIndexT2_contra,
        UserReadDataT2_co,
        UserWriteIndexT2_contra,
        UserWriteDataT2_contra,
        UserReadIndexT3_contra,
        UserReadDataT3_co,
        UserWriteIndexT3_contra,
        UserWriteDataT3_contra,
    ]
):

    backend: BackendT
    frontend: Frontend[
        BackendIndexT,
        BackendDataT,
        UserReadIndexT0_contra,
        UserReadDataT0_co,
        UserWriteIndexT0_contra,
        UserWriteDataT0_contra,
        UserReadIndexT1_contra,
        UserReadDataT1_co,
        UserWriteIndexT1_contra,
        UserWriteDataT1_contra,
        UserReadIndexT2_contra,
        UserReadDataT2_co,
        UserWriteIndexT2_contra,
        UserWriteDataT2_contra,
        UserReadIndexT3_contra,
        UserReadDataT3_co,
        UserWriteIndexT3_contra,
        UserWriteDataT3_contra,
    ]
    readonly: bool = False

    index_procs: List[IndexAdjuster[BackendIndexT]] = attrs.field(factory=list)
    read_procs: List[
        Union[
            DataProcessor[BackendDataT],
            DataWithIndexProcessor[BackendDataT, BackendIndexT],
        ]
    ] = attrs.field(factory=list)
    write_procs: List[
        Union[
            DataProcessor[BackendDataT],
            DataWithIndexProcessor[BackendDataT, BackendIndexT],
        ]
    ] = attrs.field(factory=list)

    @overload
    def read(self, idx_user: BackendIndexT) -> BackendDataT:
        ...

    @overload
    def read(self, idx_user: UserReadIndexT0_contra) -> UserReadDataT0_co:
        ...

    @overload
    def read(self, idx_user: UserReadIndexT1_contra) -> UserReadDataT1_co:
        ...

    @overload
    def read(self, idx_user: UserReadIndexT2_contra) -> UserReadDataT2_co:
        ...

    @overload
    def read(self, idx_user: UserReadIndexT3_contra) -> UserReadDataT3_co:
        ...

    def read(
        self,
        idx_user,
    ) -> Union[
        BackendDataT,
        UserReadDataT0_co,
        UserReadDataT1_co,
        UserReadDataT2_co,
        UserReadDataT3_co,
    ]:
        idx = self.frontend.convert_read_idx(idx_user)
        idx_proced = idx
        for adj in self.index_procs:
            idx_proced = adj(idx_proced)

        data_backend = self.backend.read(idx=idx_proced)

        data_backend = _apply_procs(
            data=data_backend,
            idx=idx,
            idx_proced=idx_proced,
            procs=self.read_procs,
        )
        data_user = self.frontend.convert_read_data(idx_user, data_backend)
        return data_user

    @overload
    def write(self, idx_user: BackendIndexT, data_user: BackendDataT):
        ...

    @overload
    def write(self, idx_user: UserWriteIndexT0_contra, data_user: UserWriteDataT0_contra):
        ...

    @overload
    def write(self, idx_user: UserWriteIndexT1_contra, data_user: UserWriteDataT1_contra):
        ...

    @overload
    def write(self, idx_user: UserWriteIndexT2_contra, data_user: UserWriteDataT2_contra):
        ...

    @overload
    def write(self, idx_user: UserWriteIndexT3_contra, data_user: UserWriteDataT3_contra):
        ...

    def write(
        self,
        idx_user,
        data_user,
    ):
        if self.readonly:  # pragma: no cover
            raise IOError(f"Attempting to write to a read only layer {self}")

        idx, data = self.frontend.convert_write(idx_user=idx_user, data_user=data_user)
        idx_proced = idx
        for adj in self.index_procs:
            idx_proced = adj(idx_proced)

        data = _apply_procs(
            data=data,
            idx=idx,
            idx_proced=idx_proced,
            procs=self.write_procs,
        )
        self.backend.write(idx=idx_proced, data=data)

    @property
    def name(self) -> str:  # pragma: no cover
        return self.backend.name

    @overload
    def __getitem__(self, idx_user: BackendIndexT) -> BackendDataT:
        ...

    @overload
    def __getitem__(self, idx_user: UserReadIndexT0_contra) -> UserReadDataT0_co:
        ...

    @overload
    def __getitem__(self, idx_user: UserReadIndexT1_contra) -> UserReadDataT1_co:
        ...

    @overload
    def __getitem__(self, idx_user: UserReadIndexT2_contra) -> UserReadDataT2_co:
        ...

    @overload
    def __getitem__(self, idx_user: UserReadIndexT3_contra) -> UserReadDataT3_co:
        ...

    def __getitem__(
        self, idx_user
    ) -> Union[
        BackendDataT,
        UserReadDataT0_co,
        UserReadDataT1_co,
        UserReadDataT2_co,
        UserReadDataT3_co,
    ]:  # pragma: no cover
        return self.read(idx_user)

    @overload
    def __setitem__(self, idx_user: BackendIndexT, data_user: BackendDataT):
        ...

    @overload
    def __setitem__(self, idx_user: UserWriteIndexT0_contra, data_user: UserWriteDataT0_contra):
        ...

    @overload
    def __setitem__(self, idx_user: UserWriteIndexT1_contra, data_user: UserWriteDataT1_contra):
        ...

    @overload
    def __setitem__(self, idx_user: UserWriteIndexT2_contra, data_user: UserWriteDataT2_contra):
        ...

    @overload
    def __setitem__(self, idx_user: UserWriteIndexT3_contra, data_user: UserWriteDataT3_contra):
        ...

    def __setitem__(
        self,
        idx_user,
        data_user,
    ):  # pragma: no cover
        return self.write(idx_user, data_user)

    def clone(
        self, **kwargs
    ) -> Layer[
        BackendT,
        BackendIndexT,
        BackendDataT,
        UserReadIndexT0_contra,
        UserReadDataT0_co,
        UserWriteIndexT0_contra,
        UserWriteDataT0_contra,
        UserReadIndexT1_contra,
        UserReadDataT1_co,
        UserWriteIndexT1_contra,
        UserWriteDataT1_contra,
        UserReadIndexT2_contra,
        UserReadDataT2_co,
        UserWriteIndexT2_contra,
        UserWriteDataT2_contra,
        UserReadIndexT3_contra,
        UserReadDataT3_co,
        UserWriteIndexT3_contra,
        UserWriteDataT3_contra,
    ]:  # pragma: no cover # pure delegation
        """Clones the Layer with the kwargs being passed to the backend.
        Note that `attrs.evolve` will keep the same reference to the attrs that are not
        updated, meaning that things that might be mutated must be copied and passed to it"""
        return attrs.evolve(
            self,
            backend=self.backend.clone(**kwargs),
            index_procs=copy(self.index_procs),
            read_procs=copy(self.read_procs),
            write_procs=copy(self.write_procs),
        )
