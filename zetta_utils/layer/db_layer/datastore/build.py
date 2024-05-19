# pylint: disable=missing-docstring
from __future__ import annotations

from typeguard import typechecked

from zetta_utils import builder

from .. import DBLayer, build_db_layer
from . import DatastoreBackend


@typechecked
@builder.register("build_datastore_layer")
def build_datastore_layer(
    namespace: str,
    project: str | None = None,
    database: str | None = None,
    exclude_from_indexes: tuple[str, ...] = (),
    readonly: bool = False,
) -> DBLayer:
    """Build a Datastore Layer.

    :param namespace: Namespace (zutils database) to use.
    :param project: Google Cloud project ID.
    :param database: GCP Datastore database to use.
    :param exclude_from_indexes: Tuple of column names to not be indexed.
    :param readonly: Whether layer is read only.

    :return: Datastore Layer built according to the spec.

    """

    backend = DatastoreBackend(namespace, project=project, database=database)
    backend.exclude_from_indexes = exclude_from_indexes
    result = build_db_layer(backend=backend, readonly=readonly)
    return result
