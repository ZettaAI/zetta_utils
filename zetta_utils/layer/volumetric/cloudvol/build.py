# pylint: disable=missing-docstring
from typing import Dict, Callable, Optional, Any, Iterable
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.typing import Vec3D
from zetta_utils.layer import Layer
from zetta_utils.tensor_ops import InterpolationMode

from .. import RawVolumetricIndex, VolumetricIndex, build_volumetric_layer
from . import InfoExistsModes, CVBackend, PrecomputedInfoSpec


@typechecked
@builder.register("build_cv_layer")
def build_cv_layer(  # pylint: disable=too-many-locals
    path: str,
    cv_kwargs: Optional[Dict] = None,
    default_desired_resolution: Optional[Vec3D] = None,
    index_resolution: Optional[Vec3D] = None,
    data_resolution: Optional[Vec3D] = None,
    interpolation_mode: Optional[InterpolationMode] = None,
    readonly: bool = False,
    info_reference_path: Optional[str] = None,
    info_field_overrides: Optional[Dict[str, Any]] = None,
    on_info_exists: InfoExistsModes = "expect_same",
    allow_shape_rounding: bool = False,
    index_adjs: Iterable[Callable[[VolumetricIndex], VolumetricIndex]] = (),
    read_postprocs: Iterable[Callable[..., Any]] = (),
    write_preprocs: Iterable[Callable[..., Any]] = (),
) -> Layer[
    RawVolumetricIndex, VolumetricIndex
]:  # pragma: no cover # trivial conditional, delegation only
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
    :param info_field_overrides: Manual info field specifications.
    :param on_info_exists: Behavior mode for when both new info specs aregiven
        and layer info already exists.
    :param allow_shape_rounding: Whether layer allows IO operations where the specified index
        corresponds to a non-integer number of pixels at the desired resolution. When
        ``allow_shape_rounding == True``, shapes will be rounded to nearest integer.
    :param index_adjs: List of adjustors that will be applied to the index given by the user
        prior to IO operations.
    :param read_postprocs: List of processors that will be applied to the read data before
        returning it to the user.
    :param write_preprocs: List of processors that will be applied to the data given by
        the user before writing it to the backend.
    :return: Layer built according to the spec.

    """
    if cv_kwargs is None:
        cv_kwargs = {}

    backend = CVBackend(
        path=path,
        cv_kwargs=cv_kwargs,
        on_info_exists=on_info_exists,
        info_spec=PrecomputedInfoSpec(
            reference_path=info_reference_path,
            field_overrides=info_field_overrides,
        ),
    )

    result = build_volumetric_layer(
        backend=backend,
        default_desired_resolution=default_desired_resolution,
        index_resolution=index_resolution,
        data_resolution=data_resolution,
        interpolation_mode=interpolation_mode,
        readonly=readonly,
        allow_shape_rounding=allow_shape_rounding,
        index_adjs=index_adjs,
        read_postprocs=read_postprocs,
        write_preprocs=write_preprocs,
    )
    return result
