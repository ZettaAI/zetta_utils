# pylint: disable=missing-docstring
from typing import Any, Dict, Iterable, Union

from typeguard import typechecked

from zetta_utils import builder

from .. import DataProcessor, DataWithIndexProcessor, IndexAdjuster, Layer
from . import LayerSetBackend, LayerSetFormatConverter, LayerSetIndex, UserLayerSetIndex

LayerSet = Layer[
    LayerSetIndex,  # Backend Index
    Dict[str, Any],  # Backend Data -> TODO
    UserLayerSetIndex,  # UserReadIndexT0
    Dict[str, Any],  # UserReadDataT0
    UserLayerSetIndex,  # UserWriteIndexT0
    Dict[str, Any],  # UserWriteDataT0
    ### REPEATING DEFAULTS TO FILL UP ALL TYPE VARS:
    UserLayerSetIndex,
    Dict[str, Any],
    UserLayerSetIndex,
    Dict[str, Any],
    UserLayerSetIndex,
    Dict[str, Any],
    UserLayerSetIndex,
    Dict[str, Any],
    UserLayerSetIndex,
    Dict[str, Any],
    UserLayerSetIndex,
    Dict[str, Any],
]


# from ..protocols import LayerWithIndexT, LayerWithIndexDataT


@typechecked
@builder.register("build_layer_set")
def build_layer_set(
    layers: Dict[str, Layer],
    readonly: bool = False,
    index_adjs: Iterable[IndexAdjuster[LayerSetIndex]] = (),
    read_postprocs: Iterable[
        Union[DataProcessor[Dict[str, Any]], DataWithIndexProcessor[Dict[str, Any], LayerSetIndex]]
    ] = (),
    write_preprocs: Iterable[
        Union[DataProcessor[Dict[str, Any]], DataWithIndexProcessor[Dict[str, Any], LayerSetIndex]]
    ] = (),
) -> LayerSet:
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
    backend = LayerSetBackend(layers)
    result = LayerSet(
        backend=backend,
        readonly=readonly,
        format_converter=LayerSetFormatConverter(),
        index_adjs=list(index_adjs),
        read_postprocs=list(read_postprocs),
        write_preprocs=list(write_preprocs),
    )
    return result
