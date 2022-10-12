# pylint: disable=missing-docstring
from typing import Dict, Union, Callable, Any, Sequence
from typeguard import typechecked

from zetta_utils import builder

from .. import Layer, LayerIndex, IndexAdjusterWithProcessors
from . import LayerSetBackend, SetSelectionIndex, RawSetSelectionIndex


@typechecked
@builder.register("build_layer_set")
def build_layer_set(
    layers: Dict[str, Layer],
    readonly: bool = False,
    index_adjs: Sequence[Union[Callable[..., LayerIndex], IndexAdjusterWithProcessors]] = (),
    read_postprocs: Sequence[Callable[..., Any]] = (),
    write_preprocs: Sequence[Callable[..., Any]] = (),
) -> Layer[RawSetSelectionIndex, SetSelectionIndex]:
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

    result = Layer[RawSetSelectionIndex, SetSelectionIndex](
        io_backend=backend,
        readonly=readonly,
        index_adjs=index_adjs,
        read_postprocs=read_postprocs,
        write_preprocs=write_preprocs,
    )
    return result
