import os
import sys
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from typing import Optional

import attrs

from zetta_utils import log
from zetta_utils.common import RepeatTimer
from zetta_utils.layer.db_layer import DBRowDataT, build_db_layer
from zetta_utils.layer.db_layer.datastore import DatastoreBackend
from zetta_utils.mazepa import id_generation
from zetta_utils.parsing import json

logger = log.get_logger("zetta_utils")

DEFAULT_PROJECT = "zetta-research"
RUN_DB_NAME = "run-info"
RUN_INFO_PATH = "gs://zetta_utils_runs"
RUN_DB = build_db_layer(DatastoreBackend(namespace=RUN_DB_NAME, project=DEFAULT_PROJECT))
RUN_ID = None


class RunInfo(Enum):
    ZETTA_USER = "zetta_user"
    HEARTBEAT = "heartbeat"
    CLUSTERS = "clusters"
    STATE = "state"
    TIMESTAMP = "timestamp"
    PARAMS = "params"


class RunState(Enum):
    RUNNING = "running"
    TIMEOUT = "timeout"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


def register_clusters(clusters: list) -> None:  # pragma: no cover
    """
    Register run info to database, for the garbage collector.
    """
    clusters_str = json.dumps([attrs.asdict(cluster) for cluster in clusters])
    info: DBRowDataT = {RunInfo.CLUSTERS.value: clusters_str}
    _update_run_info(info)


def _update_run_info(info: DBRowDataT) -> None:  # pragma: no cover
    row_key = f"run-{RUN_ID}"
    col_keys = tuple(info.keys())
    RUN_DB[(row_key, col_keys)] = info


@contextmanager
def run_ctx_manager(run_id: Optional[str] = None, heartbeat_interval: int = 5):
    def _send_heartbeat():
        info: DBRowDataT = {RunInfo.HEARTBEAT.value: datetime.utcnow().timestamp()}
        _update_run_info(info)

    heartbeat = None
    if run_id is None:
        run_id = id_generation.get_unique_id(slug_len=4, add_uuid=False, max_len=50)

    global RUN_ID  # pylint: disable=global-statement
    RUN_ID = run_id
    try:
        if heartbeat_interval > 0:
            heartbeat = RepeatTimer(heartbeat_interval, _send_heartbeat)
            heartbeat.start()

            # Register run only when heartbeat is enabled.
            # Auxiliary processes should not modify the main process entry.
            info: DBRowDataT = {
                RunInfo.ZETTA_USER.value: os.environ["ZETTA_USER"],
                RunInfo.TIMESTAMP.value: datetime.utcnow().timestamp(),
                RunInfo.STATE.value: RunState.RUNNING.value,
                RunInfo.PARAMS.value: " ".join(sys.argv[1:]),
            }
            _update_run_info(info)
        yield
    except Exception as e:
        _update_run_info({RunInfo.STATE.value: RunState.FAILED.value})
        raise e from None
    finally:
        RUN_ID = None
        if heartbeat:
            heartbeat.cancel()
