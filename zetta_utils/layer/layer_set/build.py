# pylint: disable=missing-docstring
from __future__ import annotations

from typing import Any, Dict, Iterable, Union

from typeguard import typechecked

from zetta_utils import builder

from .. import JointIndexDataProcessor, Layer, Processor
from . import LayerSetBackend, LayerSetFrontend, LayerSetIndex, UserLayerSetIndex

LayerSet = Layer[
    LayerSetBackend,
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
    index_procs: Iterable[Processor[LayerSetIndex]] = (),
    read_procs: Iterable[
        Union[Processor[Dict[str, Any]], JointIndexDataProcessor[Dict[str, Any], LayerSetIndex]]
    ] = (),
    write_procs: Iterable[
        Union[Processor[Dict[str, Any]], JointIndexDataProcessor[Dict[str, Any], LayerSetIndex]]
    ] = (),
) -> LayerSet:
    """Build a layer representing a set of layers given as input.

    :param layers: Mapping from layer names to layers.
    :param readonly: Whether layer is read only.
    :param index_procs: List of processors that will be applied to the index given by the user
        prior to IO operations.
    :param read_procs: List of processors that will be applied to the read data before
        returning it to the user.
    :param write_procs: List of processors that will be applied to the data given by
        the user before writing it to the backend.
    :return: Layer built according to the spec.

    """
    backend = LayerSetBackend(layers)
    result = LayerSet(
        backend=backend,
        readonly=readonly,
        frontend=LayerSetFrontend(),
        index_procs=tuple(index_procs),
        read_procs=tuple(read_procs),
        write_procs=tuple(write_procs),
    )
    return result
