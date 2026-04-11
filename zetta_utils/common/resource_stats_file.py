"""Periodic ResourceMonitor summary writer for sidecar pickup.

The main container runs this alongside the existing `monitor_resources()`
context manager. It creates its own `ResourceMonitor` instance and writes
`get_summary_stats()` output to a shared file on every tick. A sibling
sidecar process reads the file and ships the summary to Firestore.

The writer must never raise into user code — every tick is wrapped.
"""

from __future__ import annotations

import contextlib
import json
from typing import Iterator

from zetta_utils import log
from zetta_utils.common.resource_monitor import ResourceMonitor
from zetta_utils.common.timer import RepeatTimer

logger = log.get_logger("zetta_utils")

RESOURCE_STATS_FILE = "/tmp/resource_monitor_stats.json"


@contextlib.contextmanager
def write_resource_stats_file(
    interval: float | None, path: str = RESOURCE_STATS_FILE
) -> Iterator[None]:
    """Periodically sample resources and write summary to `path`.

    No-op when `interval` is None (matches `monitor_resources()` gating).

    Owns its own ResourceMonitor instance — does not share with
    `monitor_resources()`. Two instances means two psutil samples per
    interval, which is negligible compared to the cost of the worker.
    """
    if interval is None:
        yield
        return

    monitor = ResourceMonitor(log_interval_seconds=interval)

    def tick() -> None:
        try:
            monitor.log_usage()
            summary = monitor.get_summary_stats()
            if summary:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(summary, f)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(f"Failed to write resource stats file: {e}")

    timer = RepeatTimer(interval, tick)
    timer.start()
    try:
        yield
    finally:
        timer.cancel()
