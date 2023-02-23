# pylint: disable=missing-docstring
from __future__ import annotations

from typing import Iterable

from typeguard import typechecked

from zetta_utils import builder

from ... import IndexProcessor
from .. import VolumetricIndex, VolumetricLayer
from . import VolumetricLayerSet, VolumetricLayerSetBackend, VolumetricSetDataProcT


@typechecked
@builder.register("build_volumetric_layer_set")
def build_volumetric_layer_set(
    layers: dict[str, VolumetricLayer],
    readonly: bool = False,
    index_procs: Iterable[IndexProcessor[VolumetricIndex]] = (),
    read_procs: Iterable[VolumetricSetDataProcT] = (),
    write_procs: Iterable[VolumetricSetDataProcT] = (),
) -> VolumetricLayerSet:
    """Build a set of volumetric layers.
    :param layers: Mapping from layer names to layers.
    :param readonly: Whether layer is read only.
    :param index_procs: List of processors that will be applied to the index given by the user
        prior to IO operations.
    :param read_procs: List of processors that will be applied to the read data before
        returning it to the user.
    :param write_procs: List of processors that will be applied to the data given by
        the user before writing it to the backend.
    :return: Layer built according to the spec.
    """
    backend = VolumetricLayerSetBackend(layers)
    result = VolumetricLayerSet(
        backend=backend,
        readonly=readonly,
        index_procs=tuple(index_procs),
        read_procs=tuple(read_procs),
        write_procs=tuple(write_procs),
    )
    return result
