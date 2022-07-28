# pylint: disable=missing-docstring
from typing import Dict, Union, Callable, Optional, Any, Sequence, List
import copy

from zetta_utils import io
from zetta_utils import spec_parser
from zetta_utils.typing import Vec3D
from zetta_utils.io.layers import Layer
from zetta_utils.io.indexes import IndexAdjusterWithProcessors, Index
from zetta_utils.io.indexes.volumetric import AdjustDataResolution, VolumetricIndexConverter
from zetta_utils.tensor.ops import InterpolationMode


@spec_parser.register("CVLayer")
def build_cv_layer(  # pylint: disable=too-many-locals
    path: str,
    cv_kwargs: Optional[Dict] = None,
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
    if cv_kwargs is None:
        cv_kwargs = {}

    backend = io.backends.CVBackend(cloudpath=path, **cv_kwargs)
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


@spec_parser.register("LayerSet")
def build_layer_set(
    layers: Dict[str, Layer],
    readonly: bool = False,
    index_adjs: Sequence[Union[Callable[..., Index], IndexAdjusterWithProcessors]] = (),
    read_postprocs: Sequence[Callable[..., Any]] = (),
    write_preprocs: Sequence[Callable[..., Any]] = (),
) -> Layer:
    backend = io.backends.LayerSetBackend(layers)

    result = Layer(
        io_backend=backend,
        readonly=readonly,
        index_adjs=index_adjs,
        read_postprocs=read_postprocs,
        write_preprocs=write_preprocs,
    )
    return result
