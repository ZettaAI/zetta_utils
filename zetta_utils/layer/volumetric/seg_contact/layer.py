from __future__ import annotations

from collections.abc import Sequence

import attrs

from zetta_utils.layer import (
    DataProcessor,
    IndexProcessor,
    JointIndexDataProcessor,
    Layer,
)
from zetta_utils.layer.volumetric import VolumetricIndex

from .backend import SegContactLayerBackend
from .contact import SegContact

SegContactDataProcT = (
    DataProcessor[Sequence[SegContact]]
    | JointIndexDataProcessor[Sequence[SegContact], VolumetricIndex]
)


@attrs.frozen
class VolumetricSegContactLayer(
    Layer[VolumetricIndex, Sequence[SegContact], Sequence[SegContact]]
):
    """Layer for reading/writing seg_contact data."""

    backend: SegContactLayerBackend
    readonly: bool = False

    index_procs: tuple[IndexProcessor[VolumetricIndex], ...] = ()
    read_procs: tuple[SegContactDataProcT, ...] = ()
    write_procs: tuple[SegContactDataProcT, ...] = ()

    def __getitem__(self, idx: VolumetricIndex) -> Sequence[SegContact]:
        return self.read_with_procs(idx)

    def __setitem__(self, idx: VolumetricIndex, data: Sequence[SegContact]) -> None:
        if self.readonly:
            raise IOError("Cannot write to readonly layer")
        self.write_with_procs(idx, data)

    def with_changes(self, **kwargs) -> VolumetricSegContactLayer:
        """Return a new layer with the specified changes."""
        return attrs.evolve(self, **kwargs)

    def pformat(self) -> str:
        """Pretty format the layer for printing."""
        return f"VolumetricSegContactLayer(path={self.backend.path})"
