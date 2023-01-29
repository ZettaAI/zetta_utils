import os
import time

from zetta_utils.layer.db_layer import RowDataT, build_db_layer
from zetta_utils.layer.db_layer.datastore import DatastoreBackend

DEFAULT_PROJECT = "zetta-research"
EXECUTION_DB_NAME = "execution-info"
EXECUTION_DB = build_db_layer(
    DatastoreBackend(namespace=EXECUTION_DB_NAME, project=DEFAULT_PROJECT)
)


def update_execution_info(execution_id: str) -> bool:  # pragma: no cover
    execution_info: RowDataT = {
        "zetta_user": str(os.environ["ZETTA_USER"]),
        "heartbeat": time.time(),
    }

    row_key = execution_id
    col_keys = tuple(execution_info.keys())

    EXECUTION_DB[(row_key, col_keys)] = execution_info
    return True
