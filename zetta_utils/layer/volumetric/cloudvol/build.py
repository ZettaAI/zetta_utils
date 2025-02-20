# pylint: disable=missing-docstring
from __future__ import annotations

from typing import Iterable, Literal, Sequence, Union

import torch
from numpy import typing as npt

from zetta_utils import builder
from zetta_utils.geometry.bbox import BBox3D
from zetta_utils.tensor_ops import InterpolationMode

from ... import DataProcessor, IndexProcessor, JointIndexDataProcessor
from ...precomputed import InfoSpecParams, PrecomputedInfoSpec, PrecomputedVolumeDType
from .. import VolumetricIndex, VolumetricLayer, build_volumetric_layer
from . import CVBackend

# from typeguard import typechecked


# @typechecked # ypeError: isinstance() arg 2 must be a type or tuple of types on p3.9
@builder.register("build_cv_layer", versions=">=0.4")
def build_cv_layer(  # pylint: disable=too-many-locals
    path: str,
    cv_kwargs: dict | None = None,
    default_desired_resolution: Sequence[float] | None = None,
    index_resolution: Sequence[float] | None = None,
    data_resolution: Sequence[float] | None = None,
    interpolation_mode: InterpolationMode | None = None,
    readonly: bool = False,
    info_reference_path: str | None = None,
    info_inherit_all_params: bool = False,
    info_type: Literal["image", "segmentation"] | None = None,
    info_data_type: PrecomputedVolumeDType | None = None,
    info_num_channels: int | None = None,
    info_chunk_size: Sequence[int] | None = None,
    info_bbox: BBox3D | None = None,
    info_encoding: str | None = None,
    info_scales: Sequence[Sequence[float]] | None = None,
    info_extra_scale_data: dict | None = None,
    info_overwrite: bool = False,
    info_keep_existing_scales: bool = True,
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
    :param default_desired_resolution: Default resolution used when the desired
        resolution is not given as a part of an index.
    :param index_resolution: Resolution at which slices of the index will be given.
    :param data_resolution: Resolution at which data will be read from the CloudVolume
        backend. When ``data_resolution`` differs from ``desired_resolution``, data
        will be interpolated from ``data_resolution`` to ``desired_resolution`` using
        the given ``interpolation_mode``.
    :param interpolation_mode: Specification of the interpolation mode to use when
        ``data_resolution`` differs from ``desired_resolution``.
    :param readonly: Whether layer is read only.
    :param info_reference_path: Path to a reference CloudVolume for info.
    :param info_scales: List of scales to be added to the info file.
    :param info_type: Type of the volume (`image` or `segmentation`).
    :param info_data_type: Data type of the volume.
    :param info_num_channels: Number of channels of the volume.
    :param info_chunk_size: Precomputed chunk size for all new scales.
    :param info_bbox: Bounding box corresponding to the dataset bounds for all new
            scales. If `None`, will be inherited from the reference.
    :param info_encoding: Precomputed encoding for all new scales.
    :param info_extra_scale_data: Extra information to put into every scale. Not inherited
            from reference.
    :param info_inherit_all_params: Whether to inherit all unspecified parameters from the
        reference info file. If False, only the dataset bounds will be inherited.
    :param info_keep_existing_scales: Whether to keep existing scales in the info file at `path`.
    :param info_overwrite: Whether to allow overwriting existing fields/scales in the info file
        at `path.
    :param allow_slice_rounding: Whether layer allows IO operations where the specified
        index corresponds to a non-integer number of pixels at the desired resolution.
        When ``allow_slice_rounding == True``, shapes will be rounded to nearest integer.
    :param index_procs: List of processors that will be applied to the index given by
        the user prior to IO operations.
    :param read_procs: List of processors that will be applied to the read data before
        returning it to the user.
    :param write_procs: List of processors that will be applied to the data given by
        the user before writing it to the backend.
    :param cache_bytes_limit: Cache size limit in bytes.
    :return: Layer built according to the spec.
    """
    if cv_kwargs is None:
        cv_kwargs = {}

    if info_scales is not None:
        info_spec = PrecomputedInfoSpec(
            info_spec_params=InfoSpecParams.from_optional_reference(
                reference_path=info_reference_path,
                scales=info_scales,
                type=info_type,
                data_type=info_data_type,
                chunk_size=info_chunk_size,
                num_channels=info_num_channels,
                encoding=info_encoding,
                bbox=info_bbox,
                inherit_all_params=info_inherit_all_params,
                extra_scale_data=info_extra_scale_data,
            )
        )
    else:
        if any(
            param is not None
            for param in [
                info_reference_path,
                info_type,
                info_data_type,
                info_num_channels,
                info_chunk_size,
                info_bbox,
                info_encoding,
                info_extra_scale_data,
            ]
        ):
            raise ValueError(
                "When 'info_scales' is not provided, all 'info_*' parameters must be None. "
                "'info_scales' provides all the scales for the info file. An info file without "
                "scales is invalid. If 'info_scales' is not provided, the info file at the "
                "specified path is assumed to exist and will be used without modifications. "
                "Therefore, all other 'info_*' parameters must be None as they won't be used."
            )
        info_spec = PrecomputedInfoSpec(info_path=path)

    backend = CVBackend(
        path=path,
        cv_kwargs=cv_kwargs,
        info_overwrite=info_overwrite,
        info_keep_existing_scales=info_keep_existing_scales,
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
