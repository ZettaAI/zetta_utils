"""Per-run GC state Firestore accessor.

Owned exclusively by the run garbage collector. Keyed by run_id; tracks
the dominant failure class from the most recent cleanup attempt, whether
the owner has been DM'd about it, and how long the run has been blocked.
The row is cleared on full cleanup success.
"""

from __future__ import annotations

import attrs

from zetta_utils import constants
from zetta_utils.layer.db_layer.firestore import build_firestore_layer

GC_STATE_COLLECTION = "gc-run-state"
GC_STATE_DB = build_firestore_layer(
    GC_STATE_COLLECTION,
    database=constants.RUN_DATABASE,
    project=constants.DEFAULT_PROJECT,
)


@attrs.frozen
class RunGCState:
    """Per-run state owned by the garbage collector.

    :param last_error_class: Dominant failure category from the most
        recent cleanup attempt (e.g. ``"k8s_auth"``, ``"k8s_5xx"``,
        ``"sqs"``, ``"cluster_404"``). Empty when the last attempt
        succeeded.
    :param last_notify_error_class: Error class the owner was last DM'd
        about; empty if never notified. Used to suppress DMs while the
        same blocker persists.
    :param failure_cycles: Consecutive GC cycles where this run's
        cleanup failed.
    :param last_failure_ts: Unix timestamp of the most recent failure.
    :param last_attempt_ts: Unix timestamp of the most recent GC pass
        on this run.
    """

    last_error_class: str = ""
    last_notify_error_class: str = ""
    failure_cycles: int = 0
    last_failure_ts: float = 0.0
    last_attempt_ts: float = 0.0


_COLUMNS = tuple(f.name for f in attrs.fields(RunGCState))


def load_states(run_ids: list[str]) -> dict[str, RunGCState]:
    """Batched read of GC state rows for the given run ids.

    Missing rows (first-time stale runs) are returned as default
    :class:`RunGCState` instances so callers can treat them uniformly.

    :param run_ids: Run ids whose state should be loaded.
    """
    if not run_ids:
        return {}
    try:
        rows = GC_STATE_DB[(run_ids, _COLUMNS)]
    except KeyError:
        # FirestoreBackend.read prechecks the single-row case and raises
        # KeyError when the only requested row doesn't exist (multi-row
        # reads return empty dicts for missing keys). Normalize: treat
        # the missing row as a default state row.
        rows = [{} for _ in run_ids]
    return {
        run_id: RunGCState(
            last_error_class=row.get("last_error_class", ""),
            last_notify_error_class=row.get("last_notify_error_class", ""),
            failure_cycles=int(row.get("failure_cycles", 0)),
            last_failure_ts=float(row.get("last_failure_ts", 0.0)),
            last_attempt_ts=float(row.get("last_attempt_ts", 0.0)),
        )
        for run_id, row in zip(run_ids, rows)
    }


def save_state(run_id: str, state: RunGCState) -> None:
    """Write a :class:`RunGCState` row, overwriting all columns.

    :param run_id: Run id row key.
    :param state: New state to persist.
    """
    info = attrs.asdict(state)
    col_keys = tuple(info.keys())
    GC_STATE_DB[(run_id, col_keys)] = info


def clear_state(run_id: str) -> None:
    """Idempotent delete of a GC state row; called on full cleanup success.

    :param run_id: Run id row key to delete.
    """
    if run_id in GC_STATE_DB:
        del GC_STATE_DB[run_id]


def mark_notified(run_id: str, error_class: str) -> None:
    """Update only the ``last_notify_error_class`` column for ``run_id``.

    Single-column write that preserves all other persisted fields, so the
    Slack DM path can stamp "owner has been told about this error class"
    without clobbering the orchestrator's failure-cycle bookkeeping.

    :param run_id: Run id row key.
    :param error_class: Error class the owner was just notified about.
    """
    GC_STATE_DB[(run_id, ("last_notify_error_class",))] = {"last_notify_error_class": error_class}
