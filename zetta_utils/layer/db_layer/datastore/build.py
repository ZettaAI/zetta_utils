# pylint: disable=missing-docstring
from __future__ import annotations

from typing import Optional

from typeguard import typechecked

from zetta_utils import builder

from .. import DBLayer, build_db_layer
from . import DatastoreBackend


@typechecked
@builder.register("build_datastore_layer")
def build_datastore_layer(
    namespace: str,
    project: Optional[str] = None,
    readonly: bool = False,
) -> DBLayer:
    """Build a Datastore Layer.

    :param namespace: Namespace (database) to use.
    :param project: Google Cloud project ID.
    :param readonly: Whether layer is read only.

    :return: Datastore Layer built according to the spec.

    """

    backend: DatastoreBackend = DatastoreBackend(namespace, project=project)
    result = build_db_layer(backend=backend, readonly=readonly)
    return result
