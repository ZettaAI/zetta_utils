# pylint: disable=missing-docstring
from __future__ import annotations

from typing import Iterable, Sequence

from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.geometry import Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex, VolumetricLayer
from zetta_utils.layer.volumetric.layer_set.layer import (
    VolumetricLayerSet,
    VolumetricSetDataProcT,
    VolumetricSetDataWriteProcT,
)

from ... import IndexProcessor
from .backend import VolumetricSetBackend


@typechecked
@builder.register("build_volumetric_layer_set")
def build_volumetric_layer_set(
    layers: dict[str, VolumetricLayer],
    readonly: bool = False,
    default_desired_resolution: Sequence[float] | None = None,
    index_resolution: Sequence[float] | None = None,
    allow_slice_rounding: bool = False,
    index_procs: Iterable[IndexProcessor[VolumetricIndex]] = (),
    read_procs: Iterable[VolumetricSetDataProcT] = (),
    write_procs: Iterable[VolumetricSetDataWriteProcT] = (),
) -> VolumetricLayerSet:
    """
    Build a VolumetricLayerSet from a dict of VolumetricLayers.

    :param layers: Dict of VolumetricLayers to include in the set.
    :param readonly: Whether the layer is read-only.
    :param default_desired_resolution: Default resolution to use for reading data.
    :param index_resolution: Resolution to use for indexing.
    :param allow_slice_rounding: Whether to allow rounding of slice indices.
    :param index_procs: List of processors that will be applied to the index
        prior to IO operations.
    :param read_procs: List of processors that will be applied to the read data before
        returning it to the user.
    :param write_procs: List of processors that will be applied to the data given by
        the user before writing it to the backend.
    :return: Layer built according to the spec.
    """
    backend = VolumetricSetBackend(layers)

    result = VolumetricLayerSet(
        backend=backend,
        readonly=readonly,
        index_resolution=Vec3D(*index_resolution) if index_resolution else None,
        default_desired_resolution=(
            Vec3D(*default_desired_resolution) if default_desired_resolution else None
        ),
        allow_slice_rounding=allow_slice_rounding,
        index_procs=tuple(index_procs),
        read_procs=tuple(read_procs),
        write_procs=tuple(write_procs),
    )
    return result
