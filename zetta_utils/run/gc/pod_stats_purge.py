"""Manual one-shot purge of POD_STATS_DB rows for runs with stale heartbeats.

Scans POD_STATS_DB once, groups rows by ``run_id``, looks up each
run's RUN_DB heartbeat, and deletes the rows when the heartbeat is
older than ``--hours`` (default 24) or when the run has no RUN_DB row
at all. Rows missing a ``run_id`` field are treated as orphans and
deleted unconditionally. Stateless: writes nothing to ``gc-run-state``,
posts no Slack messages, mutates no collection other than POD_STATS_DB.

The main GC sweep (:mod:`zetta_utils.run.gc.orchestrator`) already
calls :func:`zetta_utils.run.finalize_run` on every fully-cleaned run,
so this tool exists only as a manual backlog drain when the steady-state
invariant has been violated.

Usage::

    python -m zetta_utils.run.gc.pod_stats_purge [--hours 24]
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import defaultdict

from zetta_utils.log import get_logger
from zetta_utils.run import strip_per_pod_resource_stats
from zetta_utils.run.db import POD_STATS_DB, RUN_DB
from zetta_utils.run.gc.utils import STALENESS_COLUMNS, is_run_stale

logger = get_logger("zetta_utils")

DEFAULT_HOURS = 24


def purge_stale_pod_stats(hours: int = DEFAULT_HOURS) -> int:
    """Delete POD_STATS_DB rows whose owning run is stale.

    A row is stale when (a) its ``run_id`` has no RUN_DB entry, (b)
    that entry's heartbeat is older than ``hours``, or (c) the row has
    no ``run_id`` field at all.

    :param hours: Staleness threshold; rows whose run heartbeat is
        older than this many hours are deleted.
    :returns: Number of POD_STATS_DB rows deleted.
    """
    threshold = time.time() - hours * 3600

    # Project only the run_id column server-side: POD_STATS_DB rows
    # carry sizable per-pod cpu / memory / gcs / semaphore payloads we
    # never look at here. Without the projection the full collection
    # would stream over the wire on every invocation.
    docs = POD_STATS_DB.query(return_columns=("run_id",))
    if not docs:
        logger.info("POD_STATS_DB is empty; nothing to purge.")
        return 0

    rows_by_run: dict[str, list[str]] = defaultdict(list)
    orphan_rows: list[str] = []
    for row_key, raw in docs.items():
        run_id = str(raw.get("run_id", ""))
        if run_id:
            rows_by_run[run_id].append(row_key)
        else:
            orphan_rows.append(row_key)

    run_ids = list(rows_by_run.keys())
    if run_ids:
        try:
            infos = RUN_DB[(run_ids, STALENESS_COLUMNS)]
        except KeyError:
            # FirestoreBackend.read prechecks the single-row case and
            # raises KeyError when the only requested row is absent;
            # normalize to empty-info defaults so the single-run case
            # behaves like the multi-row case.
            infos = [{} for _ in run_ids]
    else:
        infos = []

    to_delete: list[str] = list(orphan_rows)
    stale_runs: list[str] = []
    for run_id, info in zip(run_ids, infos):
        if is_run_stale(info, threshold):
            to_delete.extend(rows_by_run[run_id])
            stale_runs.append(run_id)

    logger.info(
        f"POD_STATS_DB: {len(docs)} rows across {len(rows_by_run)} run(s) "
        f"(+ {len(orphan_rows)} orphans without run_id). "
        f"Stale: {len(to_delete)} rows from {len(stale_runs)} run(s) "
        f"(threshold {hours}h)."
    )

    # Drop the per_pod sub-entry from each stale run's RUN_DB.resource_stats:
    # the per-pod compact summary mirrors POD_STATS_DB rows we are about to
    # delete, so leaving it on a finalized run is stale ephemera that scales
    # with the run's pod count.
    for run_id in stale_runs:
        strip_per_pod_resource_stats(run_id)

    if not to_delete:
        return 0

    del POD_STATS_DB[to_delete]
    return len(to_delete)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Delete POD_STATS_DB rows for runs whose RUN_DB heartbeat is "
            f"older than --hours (default {DEFAULT_HOURS}h)."
        )
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=DEFAULT_HOURS,
        help=f"Staleness threshold in hours (default {DEFAULT_HOURS}).",
    )
    args = parser.parse_args()
    n = purge_stale_pod_stats(hours=args.hours)
    logger.info(f"Purged {n} POD_STATS_DB rows.")


if __name__ == "__main__":  # pragma: no cover
    get_logger("zetta_utils").setLevel(logging.INFO)
    main()
