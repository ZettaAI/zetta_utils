# pylint: disable=missing-docstring
from __future__ import annotations

import copy
from typing import Iterable, Sequence

import torch
from numpy import typing as npt
from typeguard import typechecked

from zetta_utils.geometry import Vec3D
from zetta_utils.tensor_ops import InterpolationMode

from .. import DataProcessor, IndexProcessor, JointIndexDataProcessor
from . import (
    DataResolutionInterpolator,
    VolumetricBackend,
    VolumetricIndex,
    VolumetricLayer,
)


@typechecked
def build_volumetric_layer(
    backend: VolumetricBackend,
    default_desired_resolution: Sequence[float] | None = None,
    index_resolution: Sequence[float] | None = None,
    allow_slice_rounding: bool = False,
    data_resolution: Sequence[float] | None = None,
    interpolation_mode: InterpolationMode | None = None,
    readonly: bool = False,
    index_procs: Iterable[IndexProcessor[VolumetricIndex]] = (),
    read_procs: Iterable[
        DataProcessor[npt.NDArray] | JointIndexDataProcessor[npt.NDArray, VolumetricIndex]
    ] = (),
    write_procs: Iterable[
        DataProcessor[npt.NDArray | torch.Tensor]
        | JointIndexDataProcessor[torch.Tensor | npt.NDArray, VolumetricIndex]
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
    index_procs_final = copy.copy(list(index_procs))

    if data_resolution is not None:
        if interpolation_mode is None:
            raise ValueError("`data_resolution` is set, but `interpolation_mode` is not provided.")
        resolution_interpolator = DataResolutionInterpolator(
            data_resolution=data_resolution,
            interpolation_mode=interpolation_mode,
            allow_slice_rounding=allow_slice_rounding,
        )
        read_procs = [resolution_interpolator] + list(read_procs)
        write_procs = [resolution_interpolator] + list(write_procs)

    result = VolumetricLayer(
        backend=backend,
        readonly=readonly,
        index_procs=tuple(index_procs_final),
        read_procs=tuple(read_procs),
        write_procs=tuple(write_procs),
        index_resolution=Vec3D(*index_resolution) if index_resolution else None,
        default_desired_resolution=(
            Vec3D(*default_desired_resolution) if default_desired_resolution else None
        ),
        allow_slice_rounding=allow_slice_rounding,
    )
    return result
