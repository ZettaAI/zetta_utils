# pylint: disable=missing-docstring
from __future__ import annotations

from typing import Iterable, Literal, Sequence, Union

import torch
from numpy import typing as npt

from zetta_utils import builder
from zetta_utils.tensor_ops import InterpolationMode

from ... import DataProcessor, IndexProcessor, JointIndexDataProcessor
from ...precomputed import (
    InfoExistsModes,
    InfoSpecParams,
    PrecomputedInfoSpec,
    PrecomputedVolumeDType,
)
from .. import VolumetricIndex, VolumetricLayer, build_volumetric_layer
from . import TSBackend


# from typeguard import typechecked
# @typechecked
@builder.register("build_ts_layer", versions=">=0.4")
def build_ts_layer(  # pylint: disable=too-many-locals
    path: str,
    default_desired_resolution: Sequence[float] | None = None,
    index_resolution: Sequence[float] | None = None,
    data_resolution: Sequence[float] | None = None,
    interpolation_mode: InterpolationMode | None = None,
    readonly: bool = False,
    info_reference_path: str | None = None,
    info_type: Literal["image", "segmentation"] | None = None,
    info_data_type: PrecomputedVolumeDType | None = None,
    info_num_channels: int | None = None,
    info_chunk_size: Sequence[int] | None = None,
    info_dataset_size: Sequence[int] | None = None,
    info_voxel_offset: Sequence[int] | None = None,
    info_encoding: str | None = None,
    info_add_scales: Sequence[Sequence[float]] | None = None,
    inherit_all_params: bool = False,
    on_info_exists: InfoExistsModes = "expect_same",
    cache_bytes_limit: int | None = None,
    allow_slice_rounding: bool = False,
    index_procs: Iterable[IndexProcessor[VolumetricIndex]] = (),
    read_procs: Iterable[
        Union[
            DataProcessor[npt.NDArray],
            JointIndexDataProcessor[npt.NDArray, VolumetricIndex],
        ]
    ] = (),
    write_procs: Iterable[
        Union[
            DataProcessor[npt.NDArray | torch.Tensor],
            JointIndexDataProcessor[npt.NDArray | torch.Tensor, VolumetricIndex],
        ]
    ] = (),
) -> VolumetricLayer:  # pragma: no cover # trivial conditional, delegation only
    """Build a TensorStore layer.

    :param path: Path to the Precomputed volume.
    :param default_desired_resolution: Default resolution used when the desired resolution
        is not given as a part of an index.
    :param index_resolution: Resolution at which slices of the index will be given.
    :param data_resolution: Resolution at which data will be read from the CloudVolume backend.
        When ``data_resolution`` differs from ``desired_resolution``, data will be interpolated
        from ``data_resolution`` to ``desired_resolution`` using the given ``interpolation_mode``.
    :param interpolation_mode: Specification of the interpolation mode to use when
        ``data_resolution`` differs from ``desired_resolution``.
    :param readonly: Whether layer is read only.
    :param info_reference_path: Path to a reference Precomputed volume for info.
    :param info_scales: List of scales to be added to the info file.
    :param info_type: Type of the volume (`image` or `segmentation`).
    :param info_data_type: Data type of the volume.
    :param info_num_channels: Number of channels of the volume.
    :param info_chunk_size: Precomputed chunk size for all new scales.
    :param info_dataset_size: Precomputed dataset size for all new scales.
    :param info_voxel_offset: Precomputed voxel offset for all new scales.
    :param info_bounds_resolution: Resolution used to specify dataset size and voxel
        offset.
    :param info_encoding: Precomputed encoding for all new scales.
    :param inherit_all_params: Whether to inherit all unspecified parameters from the
        reference info file. If False, only the dataset bounds will be inherited.
    :param on_info_exists: Behavior mode for when both new info specs are given and
        layer info already exists.
    :param allow_slice_rounding: Whether layer allows IO operations where the specified
        index corresponds to a non-integer number of pixels at the desired resolution.
        When ``allow_slice_rounding == True``, shapes will be rounded to nearest integer.
    :param index_procs: List of processors that will be applied to the index given by
        the user prior to IO operations.
    :param read_procs: List of processors that will be applied to the read data before
        returning it to the user.
    :param write_procs: List of processors that will be applied to the data given by
        the user before writing it to the backend.
    :return: Layer built according to the spec.
    """
    if info_add_scales is not None:
        info_spec = PrecomputedInfoSpec(
            info_spec_params=InfoSpecParams.from_optional_reference(
                reference_path=info_reference_path,
                scales=info_add_scales,
                type=info_type,
                data_type=info_data_type,
                chunk_size=info_chunk_size,
                num_channels=info_num_channels,
                encoding=info_encoding,
                voxel_offset=info_voxel_offset,
                size=info_dataset_size,
                bounds_resolution=default_desired_resolution,
                inherit_all_params=inherit_all_params,
            )
        )
    else:
        info_spec = PrecomputedInfoSpec(info_path=path)

    backend = TSBackend(
        path=path,
        on_info_exists=on_info_exists,
        info_spec=info_spec,
        cache_bytes_limit=cache_bytes_limit,
    )

    result = build_volumetric_layer(
        backend=backend,
        default_desired_resolution=default_desired_resolution,
        index_resolution=index_resolution,
        data_resolution=data_resolution,
        interpolation_mode=interpolation_mode,
        readonly=readonly,
        allow_slice_rounding=allow_slice_rounding,
        index_procs=index_procs,
        read_procs=read_procs,
        write_procs=write_procs,
    )
    return result
