# pylint: disable=missing-docstring
from __future__ import annotations

from typing import Iterable

from typeguard import typechecked

from zetta_utils import builder

from .. import IndexProcessor
from . import DBBackend, DBDataProcT, DBIndex, DBLayer


@typechecked
@builder.register("build_db_layer")
def build_db_layer(
    backend: DBBackend,
    readonly: bool = False,
    index_procs: Iterable[IndexProcessor[DBIndex]] = (),
    read_procs: Iterable[DBDataProcT] = (),
    write_procs: Iterable[DBDataProcT] = (),
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
        index_procs=tuple(index_procs),
        read_procs=tuple(read_procs),
        write_procs=tuple(write_procs),
    )
    return result
