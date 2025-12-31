from __future__ import annotations

from collections.abc import Sequence

import attrs

from zetta_utils.layer import DataProcessor, IndexProcessor, JointIndexDataProcessor, Layer
from zetta_utils.layer.volumetric import VolumetricIndex

from .backend import ContactLayerBackend
from .contact import Contact

ContactDataProcT = DataProcessor[Sequence[Contact]] | JointIndexDataProcessor[
    Sequence[Contact], VolumetricIndex
]


@attrs.frozen
class VolumetricContactLayer(Layer[VolumetricIndex, Sequence[Contact], Sequence[Contact]]):
    """Layer for reading/writing contact data."""

    backend: ContactLayerBackend
    readonly: bool = False

    index_procs: tuple[IndexProcessor[VolumetricIndex], ...] = ()
    read_procs: tuple[ContactDataProcT, ...] = ()
    write_procs: tuple[ContactDataProcT, ...] = ()

    def __getitem__(self, idx: VolumetricIndex) -> Sequence[Contact]:
        raise NotImplementedError

    def __setitem__(self, idx: VolumetricIndex, data: Sequence[Contact]) -> None:
        raise NotImplementedError
