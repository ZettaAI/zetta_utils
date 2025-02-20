from contextlib import contextmanager

from zetta_utils.common import get_unique_id
from zetta_utils.constants import DEFAULT_PROJECT

from .build import build_datastore_layer


@contextmanager
def temp_db_layer_ctx(prefix: str | None = None, project: str = DEFAULT_PROJECT):
    db_layer = build_datastore_layer(
        namespace=get_unique_id(prefix=prefix, add_uuid=False), project=project
    )
    yield db_layer
    db_layer.backend.clear()
    # TODO: Delete the namespace
