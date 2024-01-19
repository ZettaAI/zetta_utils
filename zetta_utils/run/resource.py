import uuid
from enum import Enum

import attrs

from zetta_utils.layer.db_layer import build_db_layer
from zetta_utils.layer.db_layer.datastore import DatastoreBackend

DEFAULT_PROJECT = "zetta-research"
RESOURCE_DB_NAME = "run-resource"

RESOURCE_DB = build_db_layer(
    DatastoreBackend(
        namespace=RESOURCE_DB_NAME,
        project=DEFAULT_PROJECT,  # all resources must be tracked in the default project
    )
)


class ResourceTypes(Enum):
    K8S_CONFIGMAP = "k8s_configmap"
    K8S_DEPLOYMENT = "k8s_deployment"
    K8S_JOB = "k8s_job"
    K8S_SECRET = "k8s_secret"
    K8S_SERVICE = "k8s_service"
    SQS_QUEUE = "sqs_queue"


class ResourceKeys(Enum):
    RUN_ID = "run_id"
    TYPE = "type"
    NAME = "name"


@attrs.frozen
class Resource:
    run_id: str
    type: str
    name: str
    region: str = ""


def register_resource(resource: Resource) -> None:
    _resource = attrs.asdict(resource)
    row_key = str(uuid.uuid4())
    col_keys = tuple(_resource.keys())
    RESOURCE_DB[(row_key, col_keys)] = _resource
