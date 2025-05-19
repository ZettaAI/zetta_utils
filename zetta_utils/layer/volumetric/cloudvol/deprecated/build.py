# pylint: disable=missing-docstring
from __future__ import annotations

from typing import Any, Iterable, Literal, Sequence, Union

import torch
from numpy import typing as npt

from zetta_utils import builder
from zetta_utils.tensor_ops import InterpolationMode

from .... import DataProcessor, IndexProcessor, JointIndexDataProcessor
from ....deprecated.precomputed import InfoExistsModes, PrecomputedInfoSpec
from ... import VolumetricIndex, VolumetricLayer, build_volumetric_layer
from .backend import CVBackend

# from typeguard import typechecked


# @typechecked # ypeError: isinstance() arg 2 must be a type or tuple of types on p3.9
@builder.register("build_cv_layer", versions="<=0.3")
def build_cv_layer(  # pylint: disable=too-many-locals
    path: str,
    cv_kwargs: dict | None = None,
    default_desired_resolution: Sequence[float] | None = None,
    index_resolution: Sequence[float] | None = None,
    data_resolution: Sequence[float] | None = None,
    interpolation_mode: InterpolationMode | None = None,
    readonly: bool = False,
    info_extend_if_exists: bool = True,
    info_reference_path: str | None = None,
    info_type: Literal["image", "segmentation"] | None = None,
    info_data_type: str | None = None,
    info_num_channels: int | None = None,
    info_field_overrides: dict[str, Any] | None = None,
    info_chunk_size: Sequence[int] | None = None,
    info_chunk_size_map: dict[str, Sequence[int]] | None = None,
    info_dataset_size: Sequence[int] | None = None,
    info_dataset_size_map: dict[str, Sequence[int]] | None = None,
    info_voxel_offset: Sequence[int] | None = None,
    info_voxel_offset_map: dict[str, Sequence[int]] | None = None,
    info_encoding: str | None = None,
    info_encoding_map: dict[str, str] | None = None,
    info_add_scales: Sequence[Sequence[float] | dict[str, Any]] | None = None,
    info_add_scales_ref: str | dict[str, Any] | None = None,
    info_add_scales_exclude_fields: Sequence[str] = (),
    info_add_scales_mode: Literal["merge", "replace"] = "merge",
    info_only_retain_scales: Sequence[Sequence[float]] | None = None,
    on_info_exists: InfoExistsModes = "extend",
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
    cache_bytes_limit: int | None = None,
) -> VolumetricLayer:  # pragma: no cover # trivial conditional, delegation only
    """Build a CloudVolume layer.

    :param path: Path to the CloudVolume.
    :param cv_kwargs: Keyword arguments passed to the CloudVolume constructor.
    :param default_desired_resolution: Default resolution used when the desired resolution
        is not given as a part of an index.
    :param index_resolution: Resolution at which slices of the index will be given.
    :param data_resolution: Resolution at which data will be read from the CloudVolume backend.
        When ``data_resolution`` differs from ``desired_resolution``, data will be interpolated
        from ``data_resolution`` to ``desired_resolution`` using the given ``interpolation_mode``.
    :param interpolation_mode: Specification of the interpolation mode to use when
        ``data_resolution`` differs from ``desired_resolution``.
    :param readonly: Whether layer is read only.
    :param info_reference_path: Path to a reference CloudVolume for info.
    :param info_type: Type of the volume. Takes precedence over ``info_fields_overrides["type"]``.
    :param info_data_type: Data type of the volume. Takes precedence over
        ``info_fields_overrides["data_type"]``.
    :param info_num_channels: Number of channels of the volume. Takes precedence over
        ``info_fields_overrides["num_channels"]``.
    :param info_field_overrides: Manual info field specifications.
    :param info_chunk_size: Precomputed chunk size for all scales.
    :param info_chunk_size_map: Precomputed chunk size for each resolution.
    :param info_dataset_size: Precomputed dataset size for all scales.
    :param info_dataset_size_map: Precomputed dataset size for each resolution.
    :param info_voxel_offset: Precomputed voxel offset for all scales.
    :param info_voxel_offset_map: Precomputed voxel offset for each resolution.
    :param info_encoding: Precomputed encoding for all scales.
    :param info_encoding_map: Precomputed encoding for each resolution.
    :param info_add_scales: List of scales to be added based on ``info_add_scales_ref``
        Each entry can be either a resolution (e.g., [4, 4, 40]) or a partially filled
        Precomputed scale. By default, ``size`` and ``voxel_offset`` will be scaled
        accordingly to the reference scale, while keeping ``chunk_sizes`` the same.
        Note that using ``info_[chunk_size,dataset_size,voxel_offset][_map]`` will
        override these values. Using this will also sort the added and existing scales
        by their resolutions.
    :param info_add_scales_ref: Reference scale to be used. If `None`, use
        the highest available resolution scale.
    :param info_add_scales_mode: Either "merge" or "replace". "merge" will
        merge added scales to existing scales if ``info_reference_path`` is
        used, while "replace" will not keep them.
    :param info_only_retain_scales: Only keep the given scales. Evaluated after all
        other info operations except for the actual writing.
    :param on_info_exists: Behavior mode for when both new info specs aregiven
        and layer info already exists.
    :param allow_slice_rounding: Whether layer allows IO operations where the specified index
        corresponds to a non-integer number of pixels at the desired resolution. When
        ``allow_slice_rounding == True``, shapes will be rounded to nearest integer.
    :param index_procs: List of processors that will be applied to the index given by the user
        prior to IO operations.
    :param read_procs: List of processors that will be applied to the read data before
        returning it to the user.
    :param write_procs: List of processors that will be applied to the data given by
        the user before writing it to the backend.
    :param cache_bytes_limit: Cache size limit in bytes.
    :return: Layer built according to the spec.

    """
    if cv_kwargs is None:
        cv_kwargs = {}
    backend = CVBackend(
        path=path,
        cv_kwargs=cv_kwargs,
        on_info_exists=on_info_exists,
        info_spec=PrecomputedInfoSpec(
            type=info_type,
            data_type=info_data_type,
            num_channels=info_num_channels,
            reference_path=info_reference_path,
            extend_if_exists_path=path if info_extend_if_exists else None,
            field_overrides=info_field_overrides,
            default_chunk_size=info_chunk_size,
            chunk_size_map=info_chunk_size_map,
            default_dataset_size=info_dataset_size,
            dataset_size_map=info_dataset_size_map,
            default_voxel_offset=info_voxel_offset,
            voxel_offset_map=info_voxel_offset_map,
            default_encoding=info_encoding,
            encoding_map=info_encoding_map,
            add_scales=info_add_scales,
            add_scales_ref=info_add_scales_ref,
            add_scales_mode=info_add_scales_mode,
            add_scales_exclude_fields=info_add_scales_exclude_fields,
            only_retain_scales=info_only_retain_scales,
        ),
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
