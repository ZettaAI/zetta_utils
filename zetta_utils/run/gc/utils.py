"""Shared helpers for the run garbage collector (retry policy, formatters, state purge)."""

from __future__ import annotations

from typing import Any, Callable, Mapping, TypeVar

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from zetta_utils.cloud_management.resource_allocation.k8s.common import ClusterInfo
from zetta_utils.mazepa.transient_errors import TRANSIENT_ERROR_CONDITIONS
from zetta_utils.run import RunInfo, finalize_run
from zetta_utils.run.gc.state import clear_state

#: ``RUN_DB`` column projection used by every staleness gate in this
#: subpackage. Kept here so callers can stay in sync without each
#: rebuilding the tuple.
STALENESS_COLUMNS: tuple[str, str] = (RunInfo.HEARTBEAT.value, RunInfo.TIMESTAMP.value)


def _is_transient(exc: BaseException) -> bool:
    return any(cond.does_match(exc) for cond in TRANSIENT_ERROR_CONDITIONS)


#: Decorator: retries the wrapped callable up to 3 times (2s/4s/8s backoff +
#: up to 0.5s jitter) when the raised exception matches the project's canonical
#: :data:`zetta_utils.mazepa.transient_errors.TRANSIENT_ERROR_CONDITIONS`
#: catalog. Non-transient exceptions propagate immediately; the last exception
#: is re-raised after exhausting attempts.
retry_transient_api = retry(
    retry=retry_if_exception(_is_transient),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=8) + wait_random(0, 0.5),
    reraise=True,
)


T = TypeVar("T")


@retry_transient_api
def retried(call: Callable[[], T]) -> T:
    """Invoke ``call`` under the project-wide transient-API retry policy.

    Shared by every gc submodule that needs a one-shot retry around an
    arbitrary callable (deleters, discovery list calls, deregister, etc.)
    so the policy lives in one place.
    """
    return call()


def format_cluster(cluster: ClusterInfo) -> str:
    """Render a cluster as ``name (region, project)``; qualifiers omitted when absent."""
    qualifiers = [q for q in (cluster.region, cluster.project) if q]
    if not qualifiers:
        return cluster.name
    return f"{cluster.name} ({', '.join(qualifiers)})"


def format_duration(seconds: float) -> str:
    """Two-unit human-readable duration: ``2d 5h``, ``4h 13m``, ``12m 34s``, ``34s``."""
    total = max(0, int(seconds))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def is_run_stale(info: Mapping[str, Any], threshold: float) -> bool:
    """True if a run's heartbeat is older than ``threshold``.

    Falls back to ``RunInfo.TIMESTAMP`` when ``RunInfo.HEARTBEAT`` is
    missing or non-numeric; a row with neither usable field is treated
    as stale so orphan runs (no RUN_DB entry, empty info dict) are
    swept rather than silently kept.

    :param info: ``RUN_DB`` row dict; should contain
        ``RunInfo.HEARTBEAT`` and/or ``RunInfo.TIMESTAMP``.
    :param threshold: Unix-epoch staleness cutoff; heartbeats older
        than this are stale.
    """
    hb: Any = info.get(RunInfo.HEARTBEAT.value)
    if not isinstance(hb, (int, float)):
        hb = info.get(RunInfo.TIMESTAMP.value)
    if not isinstance(hb, (int, float)):
        return True
    return hb <= 0 or hb < threshold


def purge_run_state(run_id: str) -> None:
    """Consolidate final stats then drop GC-owned per-run state.

    Mirrors :func:`zetta_utils.run.finalize_run`'s aggregation (compute
    costs + aggregate pod stats + delete pod-stats rows) before dropping
    the ``gc-run-state`` row, so a run cleaned by GC ends with the same
    ``compute_cost`` / ``gcs_stats`` / ``total_egress_gib`` /
    ``semaphore_stats`` / ``resource_stats`` columns in ``RUN_DB`` as
    one that exited cleanly. Both calls are best-effort; failures are
    logged and swallowed. The run's own ``RUN_DB`` row stays as audit
    history (with ``state=TIMEDOUT`` or its prior terminal value).

    :param run_id: Run id to purge.
    """
    finalize_run(run_id)
    clear_state(run_id)
