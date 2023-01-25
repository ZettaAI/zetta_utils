# pylint: disable=missing-docstring
from __future__ import annotations

import copy
from typing import Iterable

import torch
from typeguard import typechecked
from typing_extensions import TypeAlias

from zetta_utils import builder
from zetta_utils.geometry import Vec3D
from zetta_utils.layer import Layer
from zetta_utils.tensor_ops import InterpolationMode

from .. import DataProcessor, DataWithIndexProcessor, IndexAdjuster
from . import (
    UserVolumetricIndex,
    VolumetricBackend,
    VolumetricDataInterpolator,
    VolumetricFrontend,
    VolumetricIndex,
    VolumetricIndexResolutionAdjuster,
)

VolumetricLayer: TypeAlias = Layer[
    VolumetricBackend,
    VolumetricIndex,  # Backend Index
    torch.Tensor,  # BackendData
    UserVolumetricIndex,  # UserReadIndexT0
    torch.Tensor,  # UserReadDataT0
    UserVolumetricIndex,  # UserWriteIndexT0
    torch.Tensor | float | int,  # UserWriteDataT0
    ### DUMMIES TO FILL IN
    UserVolumetricIndex,  # UserReadIndexT0
    torch.Tensor,  # UserReadDataT0
    UserVolumetricIndex,  # UserWriteIndexT0
    torch.Tensor | float | int,  # UserWriteDataT0
    UserVolumetricIndex,  # UserReadIndexT0
    torch.Tensor,  # UserReadDataT0
    UserVolumetricIndex,  # UserWriteIndexT0
    torch.Tensor | float | int,  # UserWriteDataT0
    UserVolumetricIndex,  # UserReadIndexT0
    torch.Tensor,  # UserReadDataT0
    UserVolumetricIndex,  # UserWriteIndexT0
    torch.Tensor | float | int,  # UserWriteDataT0
]


@typechecked
@builder.register("build_volumetric_layer")
def build_volumetric_layer(
    backend: VolumetricBackend,
    default_desired_resolution: Vec3D | None = None,
    index_resolution: Vec3D | None = None,
    data_resolution: Vec3D | None = None,
    interpolation_mode: InterpolationMode | None = None,
    readonly: bool = False,
    allow_slice_rounding: bool = False,
    index_procs: Iterable[IndexAdjuster[VolumetricIndex]] = (),
    read_procs: Iterable[
        DataProcessor[torch.Tensor] | DataWithIndexProcessor[torch.Tensor, VolumetricIndex]
    ] = (),
    write_procs: Iterable[
        DataProcessor[torch.Tensor] | DataWithIndexProcessor[torch.Tensor, VolumetricIndex]
    ] = (),
) -> VolumetricLayer:
    """Build a Volumetric Layer.

    :param backend: Layer backend.
    :param default_desired_resolution: Default resolution used when the desired resolution
        is not given as a part of an index.
    :param index_resolution: Resolution at which slices of the index will be given.
    :param data_resolution: Resolution at which data will be read from the CloudVolume backend.
        When ``data_resolution`` differs from ``desired_resolution``, data will be interpolated
        from ``data_resolution`` to ``desired_resolution`` using the given ``interpolation_mode``.
    :param interpolation_mode: Specification of the interpolation mode to use when
        ``data_resolution`` differs from ``desired_resolution``.
    :param readonly: Whether layer is read only.
    :param allow_slice_rounding: Whether layer allows IO operations where the specified index
        corresponds to a non-integer number of pixels at the desired resolution. When
        ``allow_slice_rounding == True``, shapes will be rounded to nearest integer.
    :param index_procs: List of processors that will be applied to the index given by the user
        prior to IO operations.
    :param read_procs: List of processors that will be applied to the read data before
        returning it to the user.
    :param write_procs: List of processors that will be applied to the data given by
        the user before writing it to the backend.
    :return: Layer built according to the spec.

    """
    frontend = VolumetricFrontend(
        index_resolution=index_resolution,
        default_desired_resolution=default_desired_resolution,
        allow_slice_rounding=allow_slice_rounding,
    )

    index_procs_final = copy.copy(list(index_procs))
    if data_resolution is not None:
        if interpolation_mode is None:
            raise ValueError("`data_resolution` is set, but `interpolation_mode` is not provided.")
        resolution_adj = VolumetricIndexResolutionAdjuster(
            resolution=data_resolution,
        )
        index_procs_final.insert(
            0,
            resolution_adj,
        )
        read_procs = list(read_procs)
        read_procs.insert(
            0,
            VolumetricDataInterpolator(
                interpolation_mode=interpolation_mode,
                mode="read",
                allow_slice_rounding=allow_slice_rounding,
            ),
        )
        write_procs = list(write_procs)
        write_procs.insert(
            -1,
            VolumetricDataInterpolator(
                interpolation_mode=interpolation_mode,
                mode="write",
                allow_slice_rounding=allow_slice_rounding,
            ),
        )

    result = VolumetricLayer(
        backend=backend,
        readonly=readonly,
        frontend=frontend,
        index_procs=tuple(index_procs_final),
        read_procs=tuple(read_procs),
        write_procs=tuple(write_procs),
    )
    return result
