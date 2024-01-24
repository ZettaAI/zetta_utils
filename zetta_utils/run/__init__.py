import os
import sys
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from typing import Optional

import attrs
import fsspec

from zetta_utils import log
from zetta_utils.common import RepeatTimer
from zetta_utils.layer.db_layer import DBRowDataT, build_db_layer
from zetta_utils.layer.db_layer.datastore import DatastoreBackend
from zetta_utils.mazepa import id_generation
from zetta_utils.parsing import json

from .resource import (
    deregister_resource,
    Resource,
    register_resource,
    ResourceTypes,
    ResourceKeys,
    RESOURCE_DB,
)

logger = log.get_logger("zetta_utils")

DEFAULT_PROJECT = "zetta-research"
RUN_INFO_BUCKET = "gs://zetta_utils_runs"
RUN_DB_NAME = "run-info"
RUN_ID = ""


class RunInfo(Enum):
    ZETTA_USER = "zetta_user"
    HEARTBEAT = "heartbeat"
    CLUSTERS = "clusters"
    STATE = "state"
    TIMESTAMP = "timestamp"
    PARAMS = "params"


RUN_DB_BACKEND = DatastoreBackend(namespace=RUN_DB_NAME, project=DEFAULT_PROJECT)
RUN_DB_BACKEND.exclude_from_indexes = (RunInfo.CLUSTERS.value, RunInfo.PARAMS.value)
RUN_DB = build_db_layer(RUN_DB_BACKEND)


class RunState(Enum):
    RUNNING = "running"
    TIMEOUT = "timeout"
    COMPLETED = "completed"
    FAILED = "failed"


def register_clusters(clusters: list) -> None:  # pragma: no cover
    """
    Register run info to database, for the garbage collector.
    """
    clusters_str = json.dumps([attrs.asdict(cluster) for cluster in clusters])
    info: DBRowDataT = {RunInfo.CLUSTERS.value: clusters_str}
    _update_run_info(info)


def record_run(spec_path: Optional[str] = None) -> None:  # pragma: no cover
    """
    Records run info in a bucket for archiving.
    """
    zetta_user = os.environ["ZETTA_USER"]
    info_path = os.environ.get("RUN_INFO_BUCKET", RUN_INFO_BUCKET)
    info_path_user = os.path.join(info_path, zetta_user)
    run_info = {
        "zetta_user": zetta_user,
        "zetta_project": os.environ["ZETTA_PROJECT"],
        "json_spec": json.loads(os.environ["ZETTA_RUN_SPEC"]),
    }
    with fsspec.open(os.path.join(info_path_user, f"{RUN_ID}.json"), "w") as f:
        json.dump(run_info, f, indent=2)

    if spec_path is not None and os.path.isfile(spec_path):
        content = None
        with open(spec_path, "r", encoding="utf-8") as src:
            content = src.read()
        with fsspec.open(os.path.join(info_path_user, f"{RUN_ID}.cue"), "w") as dst:
            dst.write(content)


def _update_run_info(info: DBRowDataT) -> None:  # pragma: no cover
    row_key = f"run-{RUN_ID}"
    col_keys = tuple(info.keys())
    RUN_DB[(row_key, col_keys)] = info


def _check_run_id_conflict(run_id: str):
    row_key = f"run-{run_id}"
    col_keys = tuple(e.value for e in RunInfo)
    if RUN_DB.exists((row_key, col_keys)):
        raise ValueError(f"RUN_ID {run_id} already exists in database.")


@contextmanager
def run_ctx_manager(run_id: Optional[str] = None, heartbeat_interval: int = 5):
    def _send_heartbeat():
        info: DBRowDataT = {RunInfo.HEARTBEAT.value: datetime.utcnow().timestamp()}
        _update_run_info(info)

    heartbeat = None
    if run_id is None:
        run_id = id_generation.get_unique_id(slug_len=4, add_uuid=False, max_len=50)
    _check_run_id_conflict(run_id)

    global RUN_ID  # pylint: disable=global-statement
    RUN_ID = run_id
    status = None
    try:
        if heartbeat_interval > 0:
            heartbeat = RepeatTimer(heartbeat_interval, _send_heartbeat)
            heartbeat.start()

            # Register run only when heartbeat is enabled.
            # Auxiliary processes should not modify the main process entry.
            status = RunState.RUNNING.value
            info: DBRowDataT = {
                RunInfo.ZETTA_USER.value: os.environ["ZETTA_USER"],
                RunInfo.TIMESTAMP.value: datetime.utcnow().timestamp(),
                RunInfo.STATE.value: status,
                RunInfo.PARAMS.value: " ".join(sys.argv[1:]),
            }
            _update_run_info(info)
        yield
    except Exception as e:
        status = RunState.FAILED.value
        raise e from None
    finally:
        if heartbeat is not None:
            _update_run_info(
                {
                    RunInfo.STATE.value: status
                    if status == RunState.FAILED.value
                    else RunState.COMPLETED.value
                }
            )
            heartbeat.cancel()
        RUN_ID = ""
