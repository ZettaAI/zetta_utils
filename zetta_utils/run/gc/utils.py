"""Shared helpers for the run garbage collector (retry policy, formatters, state purge)."""

from __future__ import annotations

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from zetta_utils.cloud_management.resource_allocation.k8s.common import ClusterInfo
from zetta_utils.mazepa.transient_errors import TRANSIENT_ERROR_CONDITIONS
from zetta_utils.run import cleanup_pod_stats
from zetta_utils.run.gc.state import clear_state


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


def purge_run_state(run_id: str) -> None:
    """Delete all GC-owned per-run state on full cleanup success.

    Drops POD_STATS_DB rows and the gc-run-state row for ``run_id``. Both
    calls are best-effort; failures are logged and swallowed. The run's
    own ``RUN_DB`` row stays as audit history (with ``state=TIMEDOUT``).

    :param run_id: Run id to purge.
    """
    cleanup_pod_stats(run_id)
    clear_state(run_id)
