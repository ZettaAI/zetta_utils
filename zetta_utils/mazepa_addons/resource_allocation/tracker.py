import os
import uuid

import attrs

from zetta_utils.layer.db_layer import build_db_layer
from zetta_utils.layer.db_layer.datastore import DatastoreBackend

EXECUTION_RESOURCE_DB_PROJECT = os.environ.get("EXECUTION_RESOURCE_DB_PROJECT", "zetta-research")

EXECUTION_RESOURCE_DB_NAME = "execution_resource"
EXECUTION_RESOURCE_DB = build_db_layer(
    DatastoreBackend(
        namespace=EXECUTION_RESOURCE_DB_NAME,
        project=EXECUTION_RESOURCE_DB_PROJECT,
    )
)


@attrs.define
class ExecutionResource:
    execution_id: str
    resource_type: str
    resource_name: str


def register_execution_resource(resource: ExecutionResource) -> None:
    resource_uuid = str(uuid.uuid4())
    EXECUTION_RESOURCE_DB[resource_uuid] = attrs.asdict(resource)  # type: ignore
