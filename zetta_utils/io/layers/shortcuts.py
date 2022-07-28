# pylint: disable=missing-docstring
from typing import Dict, Union, Callable, Optional, Any, Iterable, List

import zetta_utils as zu
from zetta_utils.io.layers import Layer
from zetta_utils.io.indexes import IndexAdjusterWithProcessors, Index
from zetta_utils.io.indexes.volumetric import AdjustDataResolution, VolumetricIndexConverter
from zetta_utils.io.backends import CVBackend
from zetta_utils.typing import Vec3D
from zetta_utils.tensor.ops import InterpolationMode


@zu.spec_parser.register("CVLayer")
def CVLayer(  # pylint: disable=invalid-name, too-many-locals
    path: str,
    cv_params: Optional[Dict] = None,
    default_desired_resolution: Optional[Vec3D] = None,
    index_resolution: Optional[Vec3D] = None,
    data_resolution: Optional[Vec3D] = None,
    readonly: bool = False,
    interpolation_mode: Optional[InterpolationMode] = None,
    index_adjs: Iterable[Union[Callable[..., Index], IndexAdjusterWithProcessors]] = (),
    read_postprocs: Iterable[Callable[..., Any]] = (),
    write_preprocs: Iterable[Callable[..., Any]] = (),
    allow_shape_rounding: bool = False,
) -> Layer:
    if cv_params is None:
        cv_params = {}
    backend = CVBackend(cloudpath=path, **cv_params)
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
