import os
import time
from datetime import datetime

from cloudfiles import CloudFiles, paths

from zetta_utils.layer.db_layer import RowDataT, build_db_layer
from zetta_utils.layer.db_layer.datastore import DatastoreBackend
from zetta_utils.log import get_logger

logger = get_logger("zetta_utils")

DEFAULT_PROJECT = "zetta-research"
EXECUTION_DB_NAME = "execution-info"
EXECUTION_INFO_PATH = "gs://zetta_utils_runs"
EXECUTION_DB = build_db_layer(
    DatastoreBackend(namespace=EXECUTION_DB_NAME, project=DEFAULT_PROJECT)
)


def add_execution_info(execution_id: str) -> None:  # pragma: no cover
    """
    Add execcution info to database.
    """
    execution_info: RowDataT = {
        "zetta_user": os.environ["ZETTA_USER"],
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


def record_execution_info(execution_id: str) -> None:  # pragma: no cover
    zetta_user = os.environ["ZETTA_USER"]
    zetta_project = os.environ["ZETTA_PROJECT"]
    zetta_run_spec_path = (os.environ.get("ZETTA_RUN_SPEC_PATH", "None"),)
    info_path = os.environ.get("EXECUTION_INFO_PATH", EXECUTION_INFO_PATH)
    info_path_user = f"{info_path}/{zetta_user}"

    execution_info = {
        "zetta_user": zetta_user,
        "zetta_project": zetta_project,
        "zetta_run_spec": os.environ["ZETTA_RUN_SPEC"],
        "zetta_run_spec_file": paths.basename(zetta_run_spec_path),
        "executed_at": datetime.utcnow(),
    }

    logger.info(f"Recording execution info to {info_path_user}/{execution_id}.")

    cf = CloudFiles(info_path_user)
    cf.put_json(execution_id, execution_info)
