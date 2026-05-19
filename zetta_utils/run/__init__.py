from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from enum import Enum
from functools import partial
from operator import itemgetter

import attrs
import fsspec
from gcsfs import GCSFileSystem
from google.cloud.firestore import DELETE_FIELD

from zetta_utils import constants, log
from zetta_utils.common import RepeatTimer, get_unique_id
from zetta_utils.layer.db_layer import DBRowDataT
from zetta_utils.layer.db_layer.firestore.backend import FirestoreBackend
from zetta_utils.parsing import json
from zetta_utils.run.costs import aggregate_pod_stats, compute_costs
from zetta_utils.run.db import POD_STATS_DB, RUN_DB

logger = log.get_logger("zetta_utils")

RUN_INFO_BUCKET = "gs://zetta_utils_runs"
RUN_ID: str | None = None

DEFAULT_HEARTBEAT_INTERVAL_SEC = 5
DEFAULT_UPDATE_COSTS_INTERVAL_SEC = int(os.environ.get("ZETTA_UPDATE_COSTS_INTERVAL_SEC", "60"))
DEFAULT_POD_STATS_INTERVAL_SEC = int(os.environ.get("ZETTA_POD_STATS_INTERVAL_SEC", "60"))


class RunInfo(Enum):
    ZETTA_USER = "zetta_user"
    HEARTBEAT = "heartbeat"
    CLUSTERS = "clusters"
    STATE = "state"
    TIMESTAMP = "timestamp"
    PARAMS = "params"
    RESULTS = "results"
    WORKER_STATE = "worker_state"
    REGION_MISMATCH = "region_mismatch"
    SEMAPHORE_WIDTHS = "semaphore_widths"


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


def update_run_results(results: dict) -> None:
    """
    Store results of a run to RUN_DB. Needs to be called when a run is active.
    """
    assert RUN_ID is not None
    info: DBRowDataT = {RunInfo.RESULTS.value: results}
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


def get_latest_checkpoint(run_id: str, zetta_user: str | None = None) -> str:
    if zetta_user is None:
        zetta_user = os.environ["ZETTA_USER"]
    info_path = os.environ.get("RUN_INFO_BUCKET", RUN_INFO_BUCKET)
    info_path_user = os.path.join(info_path, zetta_user)
    fs = GCSFileSystem(project=constants.DEFAULT_PROJECT)
    checkpoints_path = os.path.join(info_path_user, run_id)
    files = fs.ls(checkpoints_path, detail=True)
    files = [f for f in files if f["name"].endswith(".zstd")]
    files = sorted(files, key=itemgetter("ctime"), reverse=True)
    return files[0]["name"]


def update_run_info(run_id: str, info: DBRowDataT) -> None:
    col_keys = tuple(info.keys())
    RUN_DB[(run_id, col_keys)] = info


def _check_run_id_conflict():
    assert RUN_ID is not None
    if RUN_ID in RUN_DB:
        raise ValueError(f"RUN_ID {RUN_ID} already exists in database.")


def _send_heartbeat(run_id: str, bucket_egress_warned: set) -> None:
    """Send heartbeat and check for region mismatch warnings.

    Intentionally NOT wrapped in try/except: if the heartbeat write fails the
    run's k8s resources will be garbage-collected by the stale-heartbeat GC,
    so a wrap that swallows the error would let the run keep doing work it's
    about to lose. Let the failure propagate.
    """
    info: DBRowDataT = {RunInfo.HEARTBEAT.value: time.time()}
    update_run_info(run_id, info)

    error = RUN_DB[(run_id, (RunInfo.REGION_MISMATCH.value,))]
    if error:
        bucket = error[RunInfo.REGION_MISMATCH.value]["bucket"]
        message = error[RunInfo.REGION_MISMATCH.value]["message"]
        if bucket not in bucket_egress_warned:
            logger.warning(f"Region mismatch: {message}.")
            bucket_egress_warned.add(bucket)


def update_costs_safe(run_id: str | None) -> None:
    """Compute and persist costs for ``run_id``; swallow + log on failure.

    :param run_id: Run id to update; no-op when ``None``.
    """
    if run_id is None:
        return
    try:
        compute_costs(run_id)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(f"Failed to update costs: {e}")


def aggregate_pod_stats_safe(run_id: str | None) -> None:
    """Aggregate per-pod stats (gcs/semaphore/resource); swallow + log on failure.

    :param run_id: Run id to aggregate; no-op when ``None``.
    """
    if run_id is None:
        return
    try:
        aggregate_pod_stats(run_id)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(f"Failed to aggregate pod stats: {e}")


def cleanup_pod_stats(run_id: str) -> None:
    """Delete per-pod stats documents for a run in a single batched RPC.

    :param run_id: Run id whose POD_STATS_DB rows should be removed.
    """
    try:
        # Project only run_id; we just need row keys for the delete and
        # the per-pod cpu / memory / gcs / semaphore payloads would
        # otherwise stream back uselessly.
        docs = POD_STATS_DB.query(column_filter={"run_id": [run_id]}, return_columns=("run_id",))
        if docs:
            del POD_STATS_DB[list(docs.keys())]
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(f"Failed to cleanup pod stats for run {run_id}: {e}")


def strip_per_pod_resource_stats(run_id: str) -> None:
    """Delete the ``per_pod`` sub-entry from ``RUN_DB.resource_stats``.

    :func:`zetta_utils.run.costs.aggregate_pod_stats` writes a per-pod
    compact summary into ``resource_stats.per_pod`` for live inspection
    while pods are running. Once the run is finalized those pods are
    gone, so the per-pod entries become stale ephemera that grow with
    pod count; the ``fleet`` and ``per_worker_type`` rollups in
    ``resource_stats`` remain useful and are kept.

    Uses Firestore's ``DELETE_FIELD`` sentinel via a direct
    ``doc.update()`` call because :func:`update_run_info` goes through
    ``bulk_writer.set(..., merge=True)`` which can merge but cannot
    remove a sub-key. Best-effort: failures are logged and swallowed.

    :param run_id: Run id whose ``resource_stats.per_pod`` should be removed.
    """
    backend = RUN_DB.backend
    assert isinstance(backend, FirestoreBackend), (
        "strip_per_pod_resource_stats requires a FirestoreBackend for the "
        "DELETE_FIELD sentinel; got " + type(backend).__name__
    )
    try:
        ref = backend.client.collection(backend.collection).document(run_id)
        ref.update({"resource_stats.per_pod": DELETE_FIELD})
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(f"Failed to strip resource_stats.per_pod for run {run_id}: {e}")


def finalize_run(run_id: str) -> None:
    """Consolidate final cost/stats into RUN_DB then drop per-pod ephemera.

    Called from :func:`run_ctx_manager`'s ``finally`` on normal exit and
    from :mod:`zetta_utils.run.gc` when the GC fully cleans a stale or
    crashed run, so any run ends with the same consolidated columns
    (``compute_cost``, ``compute_cost_by_worker_type``, ``gcs_stats``,
    ``total_egress_gib``, ``semaphore_stats``, ``resource_stats``) in
    ``RUN_DB``. Per-pod ephemera (``POD_STATS_DB`` rows and the
    ``resource_stats.per_pod`` sub-entry) are dropped after aggregation
    so completed runs do not keep accumulating pod-scoped data.
    Idempotent and best-effort: each step swallows + logs its own
    failures so partial consolidation still makes progress.

    :param run_id: Run id to finalize.
    """
    update_costs_safe(run_id)
    aggregate_pod_stats_safe(run_id)
    cleanup_pod_stats(run_id)
    strip_per_pod_resource_stats(run_id)


def start_run_repeaters(
    run_id: str,
    bucket_egress_warned: set[str],
    heartbeat_interval: int,
    update_costs_interval: int,
    pod_stats_interval: int,
) -> list[RepeatTimer]:
    """Spawn the master's background repeaters (heartbeat, costs, pod stats).

    Returns them in the order they were started so :func:`stop_run_repeaters`
    can cancel + join uniformly. Each repeater runs in its own daemon
    thread; see :class:`zetta_utils.common.RepeatTimer`.

    :param run_id: Run id the repeaters report against.
    :param bucket_egress_warned: Mutable set passed to ``_send_heartbeat``
        to dedupe per-bucket region-mismatch warnings.
    :param heartbeat_interval: Seconds between heartbeat writes.
    :param update_costs_interval: Seconds between cost aggregations.
    :param pod_stats_interval: Seconds between pod-stats aggregations.
    """
    assert heartbeat_interval > 0
    repeaters = [
        RepeatTimer(heartbeat_interval, partial(_send_heartbeat, run_id, bucket_egress_warned)),
        RepeatTimer(update_costs_interval, partial(update_costs_safe, run_id)),
        RepeatTimer(pod_stats_interval, partial(aggregate_pod_stats_safe, run_id)),
    ]
    for repeater in repeaters:
        repeater.start()
    return repeaters


def stop_run_repeaters(repeaters: list[RepeatTimer]) -> None:
    """Cancel and join every repeater in ``repeaters``.

    Cancellation is fanned out first, then join is awaited in a second
    pass, so a slow join on one repeater doesn't keep another firing
    in the meantime.

    :param repeaters: Repeaters previously returned by
        :func:`start_run_repeaters`.
    """
    for repeater in repeaters:
        repeater.cancel()
    for repeater in repeaters:
        repeater.join()


@contextmanager
def run_ctx_manager(
    main_run_process: bool,
    run_id: str | None = None,
    spec: dict | list | None = None,
    heartbeat_interval: int = DEFAULT_HEARTBEAT_INTERVAL_SEC,
    update_costs_interval: int = DEFAULT_UPDATE_COSTS_INTERVAL_SEC,
    pod_stats_interval: int = DEFAULT_POD_STATS_INTERVAL_SEC,
):
    bucket_egress_warned: set[str] = set()

    if run_id is None:
        run_id = get_unique_id(slug_len=4, add_uuid=False, max_len=50)

    global RUN_ID  # pylint: disable=global-statement
    RUN_ID = run_id

    status = None
    assert RUN_ID is not None

    repeaters: list[RepeatTimer] = []
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

        repeaters = start_run_repeaters(
            run_id,
            bucket_egress_warned,
            heartbeat_interval,
            update_costs_interval,
            pod_stats_interval,
        )

    try:
        yield
    except BaseException:
        # BaseException so SystemExit / KeyboardInterrupt / asyncio.CancelledError
        # also mark the run as FAILED; otherwise the finally below would write
        # COMPLETED for an aborted master, leaving stale resources GC can't
        # distinguish from a clean completion.
        status = RunState.FAILED.value
        raise
    finally:
        if main_run_process:
            try:
                update_run_info(
                    RUN_ID,
                    {
                        RunInfo.STATE.value: (
                            status if status == RunState.FAILED.value else RunState.COMPLETED.value
                        )
                    },
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.warning(f"Failed to update final run state: {e}")

            stop_run_repeaters(repeaters)
            finalize_run(run_id)

        RUN_ID = None
