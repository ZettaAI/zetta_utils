# pylint: disable=missing-docstring
"""Common Layer Properties."""
from __future__ import annotations

from typing import Protocol, Tuple, TypeVar, Union, overload

BackendIndexT = TypeVar("BackendIndexT")
BackendDataT = TypeVar("BackendDataT")

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


class FormatConverter(
    Protocol[
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
    @overload
    def convert_read_idx(self, idx_user: BackendIndexT) -> BackendIndexT:
        ...

    @overload
    def convert_read_idx(self, idx_user: UserReadIndexT0_contra) -> BackendIndexT:
        ...

    @overload
    def convert_read_idx(self, idx_user: UserReadIndexT1_contra) -> BackendIndexT:
        ...

    @overload
    def convert_read_idx(self, idx_user: UserReadIndexT2_contra) -> BackendIndexT:
        ...

    @overload
    def convert_read_idx(self, idx_user: UserReadIndexT3_contra) -> BackendIndexT:
        ...

    def convert_read_idx(
        self,
        idx_user,
    ) -> BackendIndexT:
        ...

    @overload
    def convert_read_data(self, idx_user: BackendIndexT, data: BackendDataT) -> BackendDataT:
        ...

    @overload
    def convert_read_data(
        self, idx_user: UserReadIndexT0_contra, data: BackendDataT
    ) -> UserReadDataT0_co:
        ...

    @overload
    def convert_read_data(
        self, idx_user: UserReadIndexT1_contra, data: BackendDataT
    ) -> UserReadDataT1_co:
        ...

    @overload
    def convert_read_data(
        self, idx_user: UserReadIndexT2_contra, data: BackendDataT
    ) -> UserReadDataT2_co:
        ...

    @overload
    def convert_read_data(
        self, idx_user: UserReadIndexT3_contra, data: BackendDataT
    ) -> UserReadDataT3_co:
        ...

    def convert_read_data(
        self, idx_user, data
    ) -> Union[
        BackendDataT,
        UserReadDataT0_co,
        UserReadDataT1_co,
        UserReadDataT2_co,
        UserReadDataT3_co,
    ]:
        ...

    @overload
    def convert_write(
        self, idx_user: BackendIndexT, data_user: BackendDataT
    ) -> Tuple[BackendIndexT, BackendDataT]:
        ...

    @overload
    def convert_write(
        self, idx_user: UserWriteIndexT0_contra, data_user: UserWriteDataT0_contra
    ) -> Tuple[BackendIndexT, BackendDataT]:
        ...

    @overload
    def convert_write(
        self, idx_user: UserWriteIndexT1_contra, data_user: UserWriteDataT1_contra
    ) -> Tuple[BackendIndexT, BackendDataT]:
        ...

    @overload
    def convert_write(
        self, idx_user: UserWriteIndexT2_contra, data_user: UserWriteDataT2_contra
    ) -> Tuple[BackendIndexT, BackendDataT]:
        ...

    @overload
    def convert_write(
        self, idx_user: UserWriteIndexT3_contra, data_user: UserWriteDataT3_contra
    ) -> Tuple[BackendIndexT, BackendDataT]:
        ...

    def convert_write(
        self,
        idx_user,
        data_user,
    ) -> Tuple[BackendIndexT, BackendDataT]:
        ...
