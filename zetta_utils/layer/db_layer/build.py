# pylint: disable=missing-docstring
from __future__ import annotations

from typing import Iterable, List, Tuple, Union

from typeguard import typechecked
from typing_extensions import TypeAlias

from zetta_utils import builder
from zetta_utils.layer import Layer

from .. import Backend, DataProcessor, DataWithIndexProcessor, IndexAdjuster
from . import ColIndex, DataT, DBFrontend, DBIndex, RowDataT, ValueT

DBLayer: TypeAlias = Layer[
    Backend,
    DBIndex,  # Backend Index
    DataT,  # BackendData
    str,  # UserReadIndexT0
    ValueT,  # UserReadDataT0
    str,  # UserWriteIndexT0
    ValueT,  # UserWriteDataT0
    List[str],
    List[ValueT],
    List[str],
    List[ValueT],
    Tuple[str, ColIndex],
    RowDataT,
    Tuple[str, ColIndex],
    RowDataT,
    Tuple[List[str], ColIndex],
    DataT,
    Tuple[List[str], ColIndex],
    DataT,
]


@typechecked
@builder.register("build_db_layer")
def build_db_layer(
    backend: Backend[DBIndex, DataT],
    readonly: bool = False,
    index_adjs: Iterable[IndexAdjuster[DBIndex]] = (),
    read_postprocs: Iterable[
        Union[DataProcessor[DataT], DataWithIndexProcessor[DataT, DBIndex]]
    ] = (),
    write_preprocs: Iterable[
        Union[DataProcessor[DataT], DataWithIndexProcessor[DataT, DBIndex]]
    ] = (),
) -> DBLayer:
    """Build a DB Layer.

    :param backend: Layer backend.
    :param readonly: Whether layer is read only.
    :param index_adjs: List of adjustors that will be applied to the index given by the user
        prior to IO operations.
    :param read_postprocs: List of processors that will be applied to the read data before
        returning it to the user.
    :param write_preprocs: List of processors that will be applied to the data given by
        the user before writing it to the backend.
    :return: Layer built according to the spec.

    """
    result = DBLayer(
        backend=backend,
        readonly=readonly,
        frontend=DBFrontend(),
        index_adjs=list(index_adjs),
        read_postprocs=list(read_postprocs),
        write_preprocs=list(write_preprocs),
    )
    return result
