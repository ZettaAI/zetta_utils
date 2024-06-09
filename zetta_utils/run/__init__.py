from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from enum import Enum

import attrs
import fsspec

from zetta_utils import constants, log
from zetta_utils.common import RepeatTimer
from zetta_utils.layer.db_layer import DBRowDataT
from zetta_utils.layer.db_layer.firestore import build_firestore_layer
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

RUN_INFO_BUCKET = "gs://zetta_utils_runs"
COLLECTION_NAME = "run-info"
RUN_ID: str | None = None
RUN_DB = build_firestore_layer(
    COLLECTION_NAME, database=constants.RUN_DATABASE, project=constants.DEFAULT_PROJECT
)


class RunInfo(Enum):
    ZETTA_USER = "zetta_user"
    HEARTBEAT = "heartbeat"
    CLUSTERS = "clusters"
    STATE = "state"
    TIMESTAMP = "timestamp"
    PARAMS = "params"


class RunState(Enum):
    RUNNING = "running"
    TIMEDOUT = "timedout"
    COMPLETED = "completed"
    FAILED = "failed"


def register_clusters(clusters: list) -> None:
    """
    Register run info to database, for the garbage collector.
    """
    assert RUN_ID is not None
    clusters_str = json.dumps([attrs.asdict(cluster) for cluster in clusters])
    info: DBRowDataT = {RunInfo.CLUSTERS.value: clusters_str}
    update_run_info(RUN_ID, info)


def _record_run(spec: dict | list | None = None) -> None:
    """
    Records run info in a bucket for archiving.
    """
    assert RUN_ID is not None
    zetta_user = os.environ["ZETTA_USER"]
    info_path = os.environ.get("RUN_INFO_BUCKET", RUN_INFO_BUCKET)
    info_path_user = os.path.join(info_path, zetta_user)
    run_info = {
        "zetta_user": zetta_user,
        "zetta_project": os.environ["ZETTA_PROJECT"],
        "json_spec": json.dumps(spec),
    }
    with fsspec.open(os.path.join(info_path_user, f"{RUN_ID}.json"), "w") as f:
        json.dump(run_info, f, indent=2)

    if os.path.isfile(os.environ["ZETTA_RUN_SPEC_PATH"]):
        content = None
        with open(os.environ["ZETTA_RUN_SPEC_PATH"], "r", encoding="utf-8") as src:
            content = src.read()
        with fsspec.open(os.path.join(info_path_user, f"{RUN_ID}.cue"), "w") as dst:
            dst.write(content)


def update_run_info(run_id: str, info: DBRowDataT) -> None:
    row_key = f"run-{run_id}"
    col_keys = tuple(info.keys())
    RUN_DB[(row_key, col_keys)] = info


def _check_run_id_conflict():
    assert RUN_ID is not None
    row_key = f"run-{RUN_ID}"
    if row_key in RUN_DB:
        raise ValueError(f"RUN_ID {RUN_ID} already exists in database.")


@contextmanager
def run_ctx_manager(
    main_run_process: bool,
    run_id: str | None = None,
    spec: dict | list | None = None,
    heartbeat_interval: int = 5,
):
    def _send_heartbeat():
        assert RUN_ID is not None
        info: DBRowDataT = {RunInfo.HEARTBEAT.value: time.time()}
        update_run_info(RUN_ID, info)

    if run_id is None:
        run_id = id_generation.get_unique_id(slug_len=4, add_uuid=False, max_len=50)

    global RUN_ID  # pylint: disable=global-statement
    RUN_ID = run_id

    status = None
    assert RUN_ID is not None

    heartbeat_sender = None
    if main_run_process:
        _check_run_id_conflict()

        # Register run only when heartbeat is enabled.
        # Auxiliary processes should not modify the main process entry.
        status = RunState.RUNNING.value
        info: DBRowDataT = {
            RunInfo.ZETTA_USER.value: os.environ["ZETTA_USER"],
            RunInfo.TIMESTAMP.value: time.time(),
            RunInfo.STATE.value: status,
            RunInfo.PARAMS.value: " ".join(sys.argv[1:]),
        }
        _record_run(spec)
        update_run_info(RUN_ID, info)

        assert heartbeat_interval > 0
        heartbeat_sender = RepeatTimer(heartbeat_interval, _send_heartbeat)
        heartbeat_sender.start()

    try:
        yield
    except Exception as e:
        status = RunState.FAILED.value
        raise e from None
    finally:
        if main_run_process:
            update_run_info(
                RUN_ID,
                {
                    RunInfo.STATE.value: (
                        status if status == RunState.FAILED.value else RunState.COMPLETED.value
                    )
                },
            )
            assert heartbeat_sender is not None
            heartbeat_sender.cancel()
        RUN_ID = None
