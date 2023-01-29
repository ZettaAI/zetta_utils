import os
import uuid

import attrs

from zetta_utils.layer.db_layer import build_db_layer
from zetta_utils.layer.db_layer.datastore import DatastoreBackend

DEFAULT_PROJECT = "zetta-research"
EXECUTION_RESOURCE_DB_PROJECT = os.environ.get("EXECUTION_RESOURCE_DB_PROJECT", DEFAULT_PROJECT)

EXECUTION_RESOURCE_DB_NAME = f"{EXECUTION_RESOURCE_DB_PROJECT}-execution-resource"
EXECUTION_RESOURCE_DB = build_db_layer(
    DatastoreBackend(
        namespace=EXECUTION_RESOURCE_DB_NAME,
        project=DEFAULT_PROJECT,  # all resources must be tracked in the default project
    )
)


@attrs.frozen
class ExecutionResource:
    execution_id: str
    resource_type: str
    resource_name: str


def register_execution_resource(resource: ExecutionResource) -> None:
    _resource = attrs.asdict(resource)
    row_key = str(uuid.uuid4())
    col_keys = tuple(_resource.keys())
    EXECUTION_RESOURCE_DB[(row_key, col_keys)] = _resource
