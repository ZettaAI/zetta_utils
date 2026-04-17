# pylint: disable=protected-access
# mypy: disable-error-code="attr-defined,method-assign"
"""Parallel execution via forkserver with preloaded modules."""

from __future__ import annotations

import multiprocessing
import multiprocessing.forkserver
import os
from multiprocessing import resource_tracker
from typing import Callable, TypeVar

from zetta_utils.common import configure_pool_signals

MULTIPROCESSING_NUM_TASKS_THRESHOLD = 128

T = TypeVar("T")


def _patch_forkserver_reuse() -> None:  # pragma: no cover
    """Patch ForkServer.ensure_running so children reuse the parent's daemon.

    Forkserver-forked children inherit _forkserver_address and _forkserver_alive_fd
    via fd inheritance but NOT _forkserver_pid. Without this patch, ensure_running()
    sees pid=None and starts a fresh daemon, multiplying memory usage.

    This patch checks _forkserver_alive_fd: if it's set and valid, the parent's
    daemon is alive and we reuse it instead of launching a new one.

    Workaround for https://github.com/python/cpython/issues/86354
    Upstream fix: https://github.com/python/cpython/pull/139537
    """
    _orig_ensure_running = multiprocessing.forkserver.ForkServer.ensure_running

    def _patched_ensure_running(self: multiprocessing.forkserver.ForkServer) -> None:
        with self._lock:
            if self._forkserver_pid is None and self._forkserver_alive_fd is not None:
                try:
                    os.fstat(self._forkserver_alive_fd)
                except OSError:
                    self._forkserver_alive_fd = None
                    self._forkserver_address = None
                else:
                    resource_tracker.ensure_running()
                    return
        _orig_ensure_running(self)

    multiprocessing.forkserver.ForkServer.ensure_running = _patched_ensure_running


_patch_forkserver_reuse()


def get_mp_context() -> multiprocessing.context.ForkServerContext:
    """Get the forkserver multiprocessing context."""
    return multiprocessing.get_context("forkserver")


def parallel_map(fn: Callable[..., T], iterable) -> list[T]:
    """Execute fn across iterable using an ephemeral forkserver pool.

    Below MULTIPROCESSING_NUM_TASKS_THRESHOLD: runs single-threaded.
    """
    items = list(iterable)
    if len(items) <= MULTIPROCESSING_NUM_TASKS_THRESHOLD:
        return list(map(fn, items))

    with get_mp_context().Pool(initializer=configure_pool_signals) as pool:
        return pool.map(fn, items)
