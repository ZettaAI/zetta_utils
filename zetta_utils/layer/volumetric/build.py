# pylint: disable=missing-docstring
import copy
from typing import Any, Callable, Iterable, Optional

from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.layer import Layer
from zetta_utils.tensor_ops import InterpolationMode
from zetta_utils.tensor_typing import TensorTypeVar
from zetta_utils.typing import Vec3D

from .. import LayerBackend
from . import (
    RawVolumetricIndex,
    VolDataInterpolator,
    VolIdxResolutionAdjuster,
    VolumetricIndex,
    VolumetricIndexConverter,
)


@typechecked
@builder.register("build_cv_layer")
def build_volumetric_layer(
    backend: LayerBackend[VolumetricIndex, TensorTypeVar],
    default_desired_resolution: Optional[Vec3D] = None,
    index_resolution: Optional[Vec3D] = None,
    data_resolution: Optional[Vec3D] = None,
    interpolation_mode: Optional[InterpolationMode] = None,
    readonly: bool = False,
    allow_slice_rounding: bool = False,
    index_adjs: Iterable[Callable[[VolumetricIndex], VolumetricIndex]] = (),
    read_postprocs: Iterable[Callable[..., Any]] = (),
    write_preprocs: Iterable[Callable[..., Any]] = (),
) -> Layer[RawVolumetricIndex, VolumetricIndex, TensorTypeVar]:
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
    :param index_adjs: List of adjustors that will be applied to the index given by the user
        prior to IO operations.
    :param read_postprocs: List of processors that will be applied to the read data before
        returning it to the user.
    :param write_preprocs: List of processors that will be applied to the data given by
        the user before writing it to the backend.
    :return: Layer built according to the spec.

    """
    index_converter = VolumetricIndexConverter(
        index_resolution=index_resolution,
        default_desired_resolution=default_desired_resolution,
        allow_slice_rounding=allow_slice_rounding,
    )

    index_adjs_final = copy.copy(list(index_adjs))
    if data_resolution is not None:
        if interpolation_mode is None:
            raise ValueError("`data_resolution` is set, but `interpolation_mode` is not provided.")
        resolution_adj = VolIdxResolutionAdjuster(
            resolution=data_resolution,
        )
        index_adjs_final.insert(
            0,
            resolution_adj,
        )
        read_postprocs = list(read_postprocs)
        read_postprocs.insert(
            0,
            VolDataInterpolator(
                interpolation_mode=interpolation_mode,
                mode="read",
                allow_slice_rounding=allow_slice_rounding,
            ),
        )
        write_preprocs = list(write_preprocs)
        write_preprocs.insert(
            -1,
            VolDataInterpolator(
                interpolation_mode=interpolation_mode,
                mode="write",
                allow_slice_rounding=allow_slice_rounding,
            ),
        )

    result = Layer[RawVolumetricIndex, VolumetricIndex, TensorTypeVar](
        backend=backend,
        readonly=readonly,
        index_converter=index_converter,
        index_adjs=list(index_adjs_final),
        read_postprocs=list(read_postprocs),
        write_preprocs=list(write_preprocs),
    )
    return result
