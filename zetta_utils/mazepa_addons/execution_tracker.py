import os
import time

from zetta_utils.layer.db_layer import RowDataT, build_db_layer
from zetta_utils.layer.db_layer.datastore import DatastoreBackend

DEFAULT_PROJECT = "zetta-research"
EXECUTION_DB_NAME = "execution-info"
EXECUTION_DB = build_db_layer(
    DatastoreBackend(namespace=EXECUTION_DB_NAME, project=DEFAULT_PROJECT)
)


def add_execution_info(execution_id: str) -> None:  # pragma: no cover
    """
    Add execcution info to database.
    """
    execution_info: RowDataT = {
        "zetta_user": os.environ["ZETTA_USER"],
        "zetta_run_spec": os.environ["ZETTA_RUN_SPEC"],
        "heartbeat": time.time(),
    }

    row_key = execution_id
    col_keys = tuple(execution_info.keys())
    EXECUTION_DB[(row_key, col_keys)] = execution_info


def update_execution_heartbeat(execution_id: str) -> bool:  # pragma: no cover
    """
    Update execution heartbeat.
    Meant to be called periodically for upkeep (`upkeep_fn`).
    """
    execution_info: RowDataT = {
        "heartbeat": time.time(),
    }

    row_key = execution_id
    col_keys = tuple(execution_info.keys())

    EXECUTION_DB[(row_key, col_keys)] = execution_info
    return True
