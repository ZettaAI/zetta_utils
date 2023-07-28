import json
import os
import time
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from typing import Mapping

import attrs
import fsspec
from cloudfiles import paths

from zetta_utils.common import RepeatTimer
from zetta_utils.layer.db_layer import DBRowDataT, build_db_layer
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


class ExecutionInfoKeys(Enum):
    ZETTA_USER = "zetta_user"
    HEARTBEAT = "heartbeat"
    CLUSTERS = "clusters"


def register_execution(execution_id: str, clusters: list[ClusterInfo]) -> None:  # pragma: no cover
    """
    Register execution info to database, for the garbage collector.
    """
    execution_info: DBRowDataT = {
        ExecutionInfoKeys.ZETTA_USER.value: os.environ["ZETTA_USER"],
        ExecutionInfoKeys.HEARTBEAT.value: time.time(),
        ExecutionInfoKeys.CLUSTERS.value: json.dumps(
            [attrs.asdict(cluster) for cluster in clusters]
        ),
    }

    row_key = execution_id
    col_keys = tuple(execution_info.keys())
    EXECUTION_DB[(row_key, col_keys)] = execution_info


def read_execution_clusters(execution_id: str) -> list[ClusterInfo]:  # pragma: no cover
    row_key = execution_id
    col_keys = ("clusters",)
    try:
        clusters_str = EXECUTION_DB[(row_key, col_keys)][col_keys[0]]
    except KeyError:
        return []
    clusters: list[Mapping] = json.loads(clusters_str)
    return [ClusterInfo(**cluster) for cluster in clusters]


def update_execution_heartbeat(execution_id: str) -> bool:  # pragma: no cover
    """
    Update execution heartbeat.
    Meant to be called periodically for upkeep (`upkeep_fn`).
    """
    execution_info: DBRowDataT = {
        "heartbeat": time.time(),
    }

    row_key = execution_id
    col_keys = tuple(execution_info.keys())

    EXECUTION_DB[(row_key, col_keys)] = execution_info
    return True


def read_execution_run(execution_id: str) -> None:  # pragma: no cover
    """
    Reads archived execution info.
    """
    zetta_user = os.environ["ZETTA_USER"]
    info_path = os.environ.get("EXECUTION_INFO_PATH", EXECUTION_INFO_PATH)
    info_path = os.path.join(info_path, zetta_user, f"{execution_id}.json")
    logger.info(f"Reading execution info from {info_path}")

    with fsspec.open(info_path, "r") as f:
        return json.load(f)


def record_execution_run(execution_id: str) -> None:  # pragma: no cover
    """
    Records execution data in a bucket for archiving.
    """
    zetta_user = os.environ["ZETTA_USER"]
    zetta_project = os.environ["ZETTA_PROJECT"]
    zetta_run_spec_path = os.environ.get("ZETTA_RUN_SPEC_PATH", "None")

    execution_run = {
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
        json.dump(execution_run, f, indent=2)


@contextmanager
def heartbeat_tracking_ctx_mngr(execution_id, heartbeat_interval=30):
    def _send_heartbeat():
        update_execution_heartbeat(execution_id)

    heart = RepeatTimer(heartbeat_interval, _send_heartbeat)
    heart.start()
    try:
        yield
    except Exception as e:
        raise e from None
    finally:
        heart.cancel()
