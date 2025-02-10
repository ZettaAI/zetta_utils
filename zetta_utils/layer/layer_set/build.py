# pylint: disable=missing-docstring
from __future__ import annotations

from typing import Iterable, Mapping, TypeVar, Union

from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.layer.tools_base import DataProcessor, JointIndexDataProcessor

from .backend import LayerSetBackend
from .. import IndexProcessor, Layer


IndexT = TypeVar("IndexT")
DataT = TypeVar("DataT")

LayerSetDataProcT = Union[
    DataProcessor[dict[str, DataT]],
    JointIndexDataProcessor[dict[str, DataT], IndexT],
]

LayerSet = Layer[IndexT, IndexT, dict[str, DataT], dict[str, DataT]]

@builder.register("build_layer_set")
@typechecked
def build_layer_set(
    layers: Mapping[str, Layer],
    readonly: bool = False,
    index_procs: Iterable[IndexProcessor] = (),
    read_procs: Iterable[LayerSetDataProcT] = (),
    write_procs: Iterable[LayerSetDataProcT] = (),
) -> LayerSet:
    """Build a generic layer set with mix/untyped set members.
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
        index_procs=tuple(index_procs),
        read_procs=tuple(read_procs),
        write_procs=tuple(write_procs),
    )
    return result
