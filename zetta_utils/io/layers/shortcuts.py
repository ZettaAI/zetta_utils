# pylint: disable=missing-docstring
from typing import Dict, Union, Callable, Optional, Any, Iterable, List

from zetta_utils import io
from zetta_utils import spec_parser
from zetta_utils.typing import Vec3D
from zetta_utils.io.layers import Layer
from zetta_utils.io.indexes import IndexAdjusterWithProcessors, Index
from zetta_utils.io.indexes.volumetric import AdjustDataResolution, VolumetricIndexConverter
from zetta_utils.tensor.ops import InterpolationMode


@spec_parser.register("cv_layer")
def cv_layer(  # pylint: disable=too-many-locals
    path: str,
    cv_params: Optional[Dict] = None,
    default_desired_resolution: Optional[Vec3D] = None,
    index_resolution: Optional[Vec3D] = None,
    data_resolution: Optional[Vec3D] = None,
    readonly: bool = False,
    interpolation_mode: Optional[InterpolationMode] = None,
    allow_shape_rounding: bool = False,
    index_adjs: Iterable[Union[Callable[..., Index], IndexAdjusterWithProcessors]] = (),
    read_postprocs: Iterable[Callable[..., Any]] = (),
    write_preprocs: Iterable[Callable[..., Any]] = (),
) -> Layer:
    if cv_params is None:
        cv_params = {}
    backend = io.backends.CVBackend(cloudpath=path, **cv_params)
    index_converter = VolumetricIndexConverter(
        index_resolution=index_resolution,
        default_desired_resolution=default_desired_resolution,
        allow_rounding=allow_shape_rounding,
    )
    if data_resolution is not None:
        assert interpolation_mode is not None
        resolution_adj = AdjustDataResolution(
            data_resolution=data_resolution,
            interpolation_mode=interpolation_mode,
            allow_rounding=allow_shape_rounding,
        )
        index_adjs_final = [resolution_adj]  # type: List[Any]
        index_adjs_final.extend(list(index_adjs))
    else:
        index_adjs_final = list(index_adjs)

    result = Layer(
        data_backend=backend,
        readonly=readonly,
        index_converter=index_converter,
        index_adjs=index_adjs_final,
        read_postprocs=read_postprocs,
        write_preprocs=write_preprocs,
    )
    return result


@spec_parser.register("layer_set")
def layer_set(
    layers: Dict[str, Layer],
    readonly: bool = False,
    index_adjs: Iterable[Union[Callable[..., Index], IndexAdjusterWithProcessors]] = (),
    read_postprocs: Iterable[Callable[..., Any]] = (),
    write_preprocs: Iterable[Callable[..., Any]] = (),
) -> Layer:
    backend = io.backends.LayerSetBackend(layers)

    result = Layer(
        data_backend=backend,
        readonly=readonly,
        index_adjs=index_adjs,
        read_postprocs=read_postprocs,
        write_preprocs=write_preprocs,
    )
    return result
