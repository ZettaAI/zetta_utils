from contextlib import contextmanager

from zetta_utils.common import get_unique_id
from zetta_utils.constants import DEFAULT_FIRESTORE_DB, DEFAULT_PROJECT

from .build import build_firestore_layer


@contextmanager
def temp_firestore_layer_ctx(
    prefix: str | None = None, database=DEFAULT_FIRESTORE_DB, project: str = DEFAULT_PROJECT
):  # pragma: no cover # pure delegation
    db_layer = build_firestore_layer(
        collection=get_unique_id(prefix=prefix, add_uuid=False), database=database, project=project
    )
    yield db_layer
    db_layer.backend.clear()
