from contextlib import contextmanager

from zetta_utils.common import get_unique_id
from zetta_utils.constants import DEFAULT_PROJECT

from .build import build_datastore_layer


@contextmanager
def temp_datastore_layer_ctx(
    prefix: str | None = None, project: str = DEFAULT_PROJECT
):  # pragma: no cover # pure delegation
    db_layer = build_datastore_layer(
        namespace=get_unique_id(prefix=prefix, add_uuid=False), project=project
    )
    yield db_layer
    # TODO: Clear in batches to not go over the datastore limit
    db_layer.backend.clear()
