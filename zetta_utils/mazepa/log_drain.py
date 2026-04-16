"""Drain thread for forwarding worker log strings to the head process.

Workers put pre-formatted log strings on a multiprocessing.Queue.
The drain thread reads them and either prints via rich's console
(above the progress bar) or writes to stderr (when no bar is active).
"""

from __future__ import annotations

import sys
import threading
from multiprocessing import Queue
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console


class LogDrainThread(threading.Thread):
    """Daemon thread that drains pre-formatted log strings from a queue."""

    def __init__(self, queue: Queue):
        super().__init__(daemon=True)
        self.queue = queue
        self.console: Console | None = None
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                msg: str = self.queue.get(timeout=0.1)
            except Exception:
                continue
            try:
                console = self.console
                if console is not None:
                    console.print(msg, highlight=False)
                else:
                    sys.stderr.write(msg + "\n")
                    sys.stderr.flush()
            except Exception:
                pass

    def stop(self) -> None:
        self._stop_event.set()
        # Drain remaining messages.
        while True:
            try:
                msg = self.queue.get_nowait()
                console = self.console
                if console is not None:
                    console.print(msg, highlight=False)
                else:
                    sys.stderr.write(msg + "\n")
                    sys.stderr.flush()
            except Exception:
                break


# Module-level reference so progress_ctx_mngr can set/clear the console.
_active_drain: LogDrainThread | None = None


def set_drain_console(console: Console) -> None:
    if _active_drain is not None:
        _active_drain.console = console


def clear_drain_console() -> None:
    if _active_drain is not None:
        _active_drain.console = None
