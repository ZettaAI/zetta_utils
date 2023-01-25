# pylint: disable=missing-docstring
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple, Union

from typeguard import typechecked
from typing_extensions import TypeAlias

from zetta_utils import builder
from zetta_utils.layer import Layer

from .. import Backend, DataProcessor, DataWithIndexProcessor, IndexAdjuster
from . import ColIndex, DataT, DBFrontend, DBIndex, RowDataT, ValueT

DBLayer: TypeAlias = Layer[
    Backend[DBIndex, DataT],
    DBIndex,  # Backend Index
    DataT,  # BackendData
    str,  # UserReadIndexT0
    ValueT,  # UserReadDataT0
    str,  # UserWriteIndexT0
    ValueT,  # UserWriteDataT0
    List[str],
    Sequence[ValueT],
    List[str],
    Sequence[ValueT],
    Tuple[str, ColIndex],
    RowDataT,
    Tuple[str, ColIndex],
    RowDataT,
    Tuple[List[str], ColIndex],
    DataT,
    Tuple[List[str], ColIndex],
    DataT,
]

IndexProcType = IndexAdjuster[DBIndex]
ReadProcType = Union[DataProcessor[DataT], DataWithIndexProcessor[DataT, DBIndex]]
WriteProcType = ReadProcType


@typechecked
@builder.register("build_db_layer")
def build_db_layer(
    backend: Backend[DBIndex, DataT],
    readonly: bool = False,
    index_procs: Iterable[IndexProcType] = (),
    read_procs: Iterable[ReadProcType] = (),
    write_procs: Iterable[WriteProcType] = (),
) -> DBLayer:
    """Build a DB Layer.

    :param backend: Layer backend.
    :param readonly: Whether layer is read only.
    :param index_procs: List of processors that will be applied to the index given by the user
        prior to IO operations.
    :param read_procs: List of processors that will be applied to the read data before
        returning it to the user.
    :param write_procs: List of processors that will be applied to the data given by
        the user before writing it to the backend.
    :return: Layer built according to the spec.

    """
    result = DBLayer(
        backend=backend,
        readonly=readonly,
        frontend=DBFrontend(),
        index_procs=tuple(index_procs),
        read_procs=tuple(read_procs),
        write_procs=tuple(write_procs),
    )
    return result
