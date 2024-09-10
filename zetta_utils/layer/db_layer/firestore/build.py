# pylint: disable=missing-docstring
from __future__ import annotations

from typeguard import typechecked

from zetta_utils import builder

from .. import DBLayer, build_db_layer
from . import FirestoreBackend


@typechecked
@builder.register("build_firestore_layer")
def build_firestore_layer(
    collection: str,
    database: str | None = None,
    project: str | None = None,
    readonly: bool = False,
) -> DBLayer:
    """Build a Firestore Layer.

    :param collection: is a group of related documents/rows in firestore.
    :param database: GCP Firestore database to use.
    :param project: Google Cloud project ID.
    :param readonly: Whether layer is read only.

    :return: Datastore Layer built according to the spec.

    """

    backend = FirestoreBackend(collection, database=database, project=project)
    result = build_db_layer(backend=backend, readonly=readonly)
    return result
