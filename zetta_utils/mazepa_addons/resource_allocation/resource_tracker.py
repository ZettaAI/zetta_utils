import uuid
from enum import Enum

import attrs

from zetta_utils.layer.db_layer import build_db_layer
from zetta_utils.layer.db_layer.datastore import DatastoreBackend

DEFAULT_PROJECT = "zetta-research"
EXECUTION_RESOURCE_DB_NAME = "execution-resource"

EXECUTION_RESOURCE_DB = build_db_layer(
    DatastoreBackend(
        namespace=EXECUTION_RESOURCE_DB_NAME,
        project=DEFAULT_PROJECT,  # all resources must be tracked in the default project
    )
)


class ExecutionResourceTypes(Enum):
    K8S_DEPLOYMENT = "k8s_deployment"
    K8S_SECRET = "k8s_secret"
    SQS_QUEUE = "sqs_queue"


class ExecutionResourceKeys(Enum):
    EXECUTION_ID = "execution_id"
    TYPE = "type"
    NAME = "name"


@attrs.frozen
class ExecutionResource:
    execution_id: str
    type: str
    name: str


def register_execution_resource(resource: ExecutionResource) -> None:
    _resource = attrs.asdict(resource)
    row_key = str(uuid.uuid4())
    col_keys = tuple(_resource.keys())
    EXECUTION_RESOURCE_DB[(row_key, col_keys)] = _resource
