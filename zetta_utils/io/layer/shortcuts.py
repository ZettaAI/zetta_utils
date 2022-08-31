# pylint: disable=missing-docstring
from typing import Dict, Union, Callable, Optional, Any, Sequence, List
import copy
from typeguard import typechecked

from zetta_utils import io
from zetta_utils import builder
from zetta_utils.typing import Vec3D
from zetta_utils.io.layer import Layer
from zetta_utils.io.indexes import IndexAdjusterWithProcessors, Index
from zetta_utils.io.indexes.volumetric import AdjustDataResolution, VolumetricIndexConverter
from zetta_utils.tensor_ops import InterpolationMode


@typechecked
@builder.register("CVLayer")
def build_cv_layer(  # pylint: disable=too-many-locals
    path: str,
    cv_kwargs: Optional[Dict] = None,
    device: str = "cpu",
    default_desired_resolution: Optional[Vec3D] = None,
    index_resolution: Optional[Vec3D] = None,
    data_resolution: Optional[Vec3D] = None,
    interpolation_mode: Optional[InterpolationMode] = None,
    readonly: bool = False,
    allow_shape_rounding: bool = False,
    index_adjs: Sequence[Union[Callable[..., Index], IndexAdjusterWithProcessors]] = (),
    read_postprocs: Sequence[Callable[..., Any]] = (),
    write_preprocs: Sequence[Callable[..., Any]] = (),
) -> Layer:
    """Build a CloudVolume layer.

    :param path: Path to the CloudVolume.
    :param cv_kwargs: Keyword arguments passed to the CloudVolume constructor.
    :param device: Device name on which read tensors will reside.
    :param default_desired_resolution: Default resolution used when the desired resolution
        is not given as a part of an index.
    :param index_resolution: Resolution at which slices of the index will be given.
    :param data_resolution: Resolution at which data will be read from the CloudVolume backend.
        When ``data_resolution`` differs from ``desired_resolution``, data will be interpolated
        from ``data_resolution`` to ``desired_resolution`` using the given ``interpolation_mode``.
    :param interpolation_mode: Specification of the interpolation mode to use when
        ``data_resolution`` differs from ``desired_resolution``.
    :param readonly: Whether layer is read only.
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

    backend = io.backends.CVBackend(cloudpath=path, device=device, **cv_kwargs)
    index_converter = VolumetricIndexConverter(
        index_resolution=index_resolution,
        default_desired_resolution=default_desired_resolution,
        allow_rounding=allow_shape_rounding,
    )
    if data_resolution is not None:
        if interpolation_mode is None:
            raise ValueError("`data_resolution` is set, but `interpolation_mode` is not provided.")
        resolution_adj = AdjustDataResolution(
            data_resolution=data_resolution,
            interpolation_mode=interpolation_mode,
            allow_rounding=allow_shape_rounding,
        )
        index_adjs_final = [resolution_adj]  # type: List[Any]
        index_adjs_final.extend(list(index_adjs))
    else:
        index_adjs_final = copy.copy(list(index_adjs))

    result = Layer(
        io_backend=backend,
        readonly=readonly,
        index_converter=index_converter,
        index_adjs=index_adjs_final,
        read_postprocs=read_postprocs,
        write_preprocs=write_preprocs,
    )
    return result


@typechecked
@builder.register("LayerSet")
def build_layer_set(
    layers: Dict[str, Layer],
    readonly: bool = False,
    index_adjs: Sequence[Union[Callable[..., Index], IndexAdjusterWithProcessors]] = (),
    read_postprocs: Sequence[Callable[..., Any]] = (),
    write_preprocs: Sequence[Callable[..., Any]] = (),
) -> Layer:
    """Build a layer representing a set of layers given as input.

    :param layers: Mapping from layer names to layers.
    :param readonly: Whether layer is read only.
    :param index_adjs: List of adjustors that will be applied to the index given by the user
        prior to IO operations.
    :param read_postprocs: List of processors that will be applied to the read data before
        returning it to the user.
    :param write_preprocs: List of processors that will be applied to the data given by
        the user before writing it to the backend.
    :return: Layer built according to the spec.

    """
    backend = io.backends.LayerSetBackend(layers)

    result = Layer(
        io_backend=backend,
        readonly=readonly,
        index_adjs=index_adjs,
        read_postprocs=read_postprocs,
        write_preprocs=write_preprocs,
    )
    return result
