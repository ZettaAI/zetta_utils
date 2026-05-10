"""Shared helpers for the run garbage collector (retry policy, state purge)."""

from __future__ import annotations

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

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


def purge_run_state(run_id: str) -> None:
    """Delete all GC-owned per-run state on full cleanup success.

    Drops POD_STATS_DB rows and the gc-run-state row for ``run_id``. Both
    calls are best-effort; failures are logged and swallowed. The run's
    own ``RUN_DB`` row stays as audit history (with ``state=TIMEDOUT``).

    :param run_id: Run id to purge.
    """
    cleanup_pod_stats(run_id)
    clear_state(run_id)
