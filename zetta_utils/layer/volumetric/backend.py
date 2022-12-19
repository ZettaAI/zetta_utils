# pylint: disable=missing-docstring # pragma: no cover
from abc import abstractmethod

import torch

from .. import Backend
from . import VolumetricIndex


class VolumetricBackend(
    Backend[VolumetricIndex, torch.Tensor]
):  # pylint: disable=too-few-public-methods
    @property
    @abstractmethod
    def enforce_chunk_aligned_writes(self) -> bool:
        ...

    @enforce_chunk_aligned_writes.setter
    @abstractmethod
    def enforce_chunk_aligned_writes(self, value: bool) -> None:
        ...
