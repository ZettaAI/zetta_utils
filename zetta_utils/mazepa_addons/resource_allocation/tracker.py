import attrs

from zetta_utils.layer.db_layer import DBLayer, build_db_layer
from zetta_utils.layer.db_layer.datastore import DatastoreBackend

EXECUTION_RESOURCE_DB = "execution_resource"


@attrs.define
class ExecutionResource:
    execution_id: str
    resource_type: str
    resource_name: str


def get_execution_resource_db() -> DBLayer:
    backend = DatastoreBackend(EXECUTION_RESOURCE_DB)
    return build_db_layer(backend)
