"""Temporary backlog drain: strip ``resource_stats.per_pod`` from stale runs.

Scans RUN_DB, identifies runs whose heartbeat is older than ``--hours``
(default 24), and removes the ``resource_stats.per_pod`` sub-entry from
each via batched Firestore writes. Intended as a one-shot to clean up
backlog that accumulated before :func:`zetta_utils.run.finalize_run`
started stripping ``per_pod`` automatically; the steady-state path no
longer needs this tool.

Updates are issued through a ``BulkWriter`` in batches of
``--batch-size`` (default 500) so the run-time is dominated by
Firestore throughput rather than per-call round trips. Processed run
ids are appended to a checkpoint file after each batch flush so an
interrupted invocation re-runs at most one in-flight batch; the
``DELETE_FIELD`` sentinel is idempotent on already-missing fields, so
the replay is safe.

Usage::

    python -m zetta_utils.run.gc.per_pod_purge [--hours 24] \\
        [--checkpoint per_pod_purge_checkpoint.txt] \\
        [--batch-size 500]
"""

from __future__ import annotations

import argparse
import logging
import time

from google.cloud.firestore import DELETE_FIELD

from zetta_utils.layer.db_layer.firestore.backend import FirestoreBackend
from zetta_utils.log import get_logger
from zetta_utils.run.db import RUN_DB
from zetta_utils.run.gc.utils import STALENESS_COLUMNS, is_run_stale

logger = get_logger("zetta_utils")

DEFAULT_HOURS = 24
DEFAULT_CHECKPOINT = "per_pod_purge_checkpoint.txt"

#: BulkWriter default batch size. Firestore's documented max-ops per
#: batch is 500; sticking to that maximizes throughput while leaving
#: room for the SDK's internal flow control to retry without blowing
#: the per-RPC ceiling.
DEFAULT_BATCH_SIZE = 500


def _load_checkpoint(path: str) -> set[str]:
    """Read processed run ids from ``path``; return empty set if absent."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    except FileNotFoundError:
        return set()


def _append_checkpoint(path: str, run_ids: list[str]) -> None:
    """Append a batch of run ids; flush + close so a SIGKILL after the
    write cannot lose the just-completed batch.
    """
    with open(path, "a", encoding="utf-8") as f:
        f.writelines(run_id + "\n" for run_id in run_ids)
        f.flush()


def purge_stale_per_pod(
    hours: int = DEFAULT_HOURS,
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Strip ``resource_stats.per_pod`` from every run with stale heartbeat.

    Streams RUN_DB once with a heartbeat+timestamp projection, filters
    to runs older than the threshold, drops any already in the
    checkpoint, then issues batched ``DELETE_FIELD`` updates through a
    :class:`google.cloud.firestore.BulkWriter`. After each batch
    flushes successfully, the batch's run ids are appended to the
    checkpoint file.

    :param hours: Staleness threshold; runs whose heartbeat is older
        than this many hours have their ``per_pod`` entry stripped.
    :param checkpoint_path: File path where processed run ids are
        appended after each batch flush. Re-running with the same path
        skips runs already recorded there; delete the file to start
        over.
    :param batch_size: Number of ``DELETE_FIELD`` updates per
        BulkWriter flush. Defaults to Firestore's per-batch max (500).
    :returns: Number of runs the strip was attempted on this
        invocation (excludes runs already in the checkpoint).
    """
    threshold = time.time() - hours * 3600
    processed = _load_checkpoint(checkpoint_path)

    runs = RUN_DB.query(return_columns=STALENESS_COLUMNS)
    stale_run_ids: list[str] = [
        run_id for run_id, info in runs.items() if is_run_stale(info, threshold)
    ]

    pending = [run_id for run_id in stale_run_ids if run_id not in processed]
    logger.info(
        f"RUN_DB: scanned {len(runs)} run(s); {len(stale_run_ids)} stale "
        f"candidates (threshold {hours}h); "
        f"{len(stale_run_ids) - len(pending)} already in checkpoint "
        f"`{checkpoint_path}`; {len(pending)} to strip in batches of {batch_size}."
    )

    backend = RUN_DB.backend
    assert isinstance(backend, FirestoreBackend), (
        "per_pod_purge requires a FirestoreBackend for the DELETE_FIELD sentinel; "
        "got " + type(backend).__name__
    )
    client = backend.client
    coll = client.collection(backend.collection)

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        bulk = client.bulk_writer()
        for run_id in batch:
            bulk.update(coll.document(run_id), {"resource_stats.per_pod": DELETE_FIELD})
        bulk.flush()
        _append_checkpoint(checkpoint_path, batch)
        end = min(start + batch_size, len(pending))
        logger.info(f"...stripped {end}/{len(pending)} run(s).")
    return len(pending)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Strip resource_stats.per_pod from runs whose RUN_DB heartbeat is "
            f"older than --hours (default {DEFAULT_HOURS}h), using batched "
            "Firestore writes. Resumable via the checkpoint file."
        )
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=DEFAULT_HOURS,
        help=f"Staleness threshold in hours (default {DEFAULT_HOURS}).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help=(
            "Path to the checkpoint file; processed run ids are appended "
            f"here after each batch flush (default {DEFAULT_CHECKPOINT}). "
            "Delete the file to start over."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "How many DELETE_FIELD updates to send per BulkWriter flush "
            f"(default {DEFAULT_BATCH_SIZE}, Firestore's per-batch max). "
            "Lower if you hit throttling."
        ),
    )
    args = parser.parse_args()
    n = purge_stale_per_pod(
        hours=args.hours,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
    )
    logger.info(f"Stripped resource_stats.per_pod from {n} run(s) this invocation.")


if __name__ == "__main__":  # pragma: no cover
    get_logger("zetta_utils").setLevel(logging.INFO)
    main()
