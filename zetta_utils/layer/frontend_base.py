# pylint: disable=missing-docstring # pragma: no cover
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

UserIndexT = TypeVar("UserIndexT")
IndexT = TypeVar("IndexT")
UserDataT = TypeVar("UserDataT")
DataT = TypeVar("DataT")


class Frontend(ABC, Generic[UserIndexT, IndexT, UserDataT, DataT]):  # pylint: disable=too-few-public-methods
    @abstractmethod
    def convert_idx(self, idx_user: UserIndexT) -> IndexT:
        """Converts user index to backend index"""

    @abstractmethod
    def convert_write(
        self,
        idx_user: UserIndexT,
        data_user: DataT | UserDataT,
    ) -> tuple[IndexT, DataT]:
        """Converts user index and data to backend index and data"""
        
class NoopFrontend(Frontend[IndexT, IndexT, DataT, DataT]):
    def convert_idx(self, idx_user: IndexT) -> IndexT:
        return idx_user

    @abstractmethod
    def convert_write(
        self,
        idx_user: IndexT,
        data_user: DataT ,
    ) -> tuple[IndexT, DataT]:  
        return (idx_user, data_user)
