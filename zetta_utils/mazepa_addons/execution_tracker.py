import json
import os
import time
from datetime import datetime
from typing import Mapping

import attrs
import fsspec
from cloudfiles import paths

from zetta_utils.layer.db_layer import RowDataT, build_db_layer
from zetta_utils.layer.db_layer.datastore import DatastoreBackend
from zetta_utils.log import get_logger

from .resource_allocation.k8s import ClusterInfo

logger = get_logger("zetta_utils")

DEFAULT_PROJECT = "zetta-research"
EXECUTION_DB_NAME = "execution-info"
EXECUTION_INFO_PATH = "gs://zetta_utils_runs"
EXECUTION_DB = build_db_layer(
    DatastoreBackend(namespace=EXECUTION_DB_NAME, project=DEFAULT_PROJECT)
)


def register_execution(execution_id: str, clusters: list[ClusterInfo]) -> None:  # pragma: no cover
    """
    Register execution info to database, for the garbage collector.
    """
    execution_info: RowDataT = {
        "zetta_user": os.environ["ZETTA_USER"],
        "heartbeat": time.time(),
        "clusters": json.dumps([attrs.asdict(cluster) for cluster in clusters]),
    }

    row_key = execution_id
    col_keys = tuple(execution_info.keys())
    EXECUTION_DB[(row_key, col_keys)] = execution_info


def read_execution_clusters(execution_id: str) -> list[ClusterInfo]:
    row_key = execution_id
    col_keys = ("clusters",)
    clusters_str = EXECUTION_DB[(row_key, col_keys)][col_keys[0]]
    clusters: list[Mapping] = json.loads(clusters_str)  # type: ignore
    return [ClusterInfo(**cluster) for cluster in clusters]


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


def record_execution_run(execution_id: str) -> None:  # pragma: no cover
    """
    Records execution data in a bucket for archiving.
    """
    zetta_user = os.environ["ZETTA_USER"]
    zetta_project = os.environ["ZETTA_PROJECT"]
    zetta_run_spec_path = os.environ.get("ZETTA_RUN_SPEC_PATH", "None")

    execution_info = {
        "zetta_user": zetta_user,
        "zetta_project": zetta_project,
        "executed_ts": datetime.utcnow().isoformat(),
        "zetta_run_spec_file": paths.basename(zetta_run_spec_path),
        "zetta_run_spec": json.loads(os.environ["ZETTA_RUN_SPEC"]),
    }

    info_path = os.environ.get("EXECUTION_INFO_PATH", EXECUTION_INFO_PATH)
    info_path = os.path.join(info_path, zetta_user, f"{execution_id}.json")
    logger.info(f"Recording execution info to {info_path}")

    with fsspec.open(info_path, "w") as f:
        json.dump(execution_info, f, indent=2)
