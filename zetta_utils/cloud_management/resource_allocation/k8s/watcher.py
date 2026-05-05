"""
Generic infrastructure for k8s watchers: api client cache, resilient
``watch.Watch`` loop with reset-on-error + batched logging, and the
:class:`BatchedWarner` summary helper used by both content-event watchers
and the resilient loop's transient-error throttle.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Callable

import kubernetes.client as k8s_client
from kubernetes.client.exceptions import ApiException

from kubernetes import watch  # type: ignore
from zetta_utils import log

from .common import ClusterInfo, get_cluster_data

logger = log.get_logger("zetta_utils")


def open_watcher_log(log_dir: str | None, run_id: str, filename: str) -> Path | None:
    if not log_dir:
        return None
    out_dir = Path(log_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename


def append_watcher_log(log_path: Path | None, msg: str) -> None:
    if log_path is None:
        return
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


_core_v1_api: k8s_client.CoreV1Api | None = None


def get_core_v1_api(cluster_info: ClusterInfo) -> k8s_client.CoreV1Api:
    """Process-wide ``CoreV1Api`` shared across the watcher threads.

    Built from ``get_cluster_data(cluster_info)`` (programmatic ADC auth)
    so it works without a local kubeconfig. Reset on watcher failure so
    the next reconnect rebuilds against fresh credentials.
    """
    global _core_v1_api  # pylint: disable=global-statement
    if _core_v1_api is None:
        configuration, _ = get_cluster_data(cluster_info)
        k8s_client.Configuration.set_default(configuration)
        _core_v1_api = k8s_client.CoreV1Api()
    return _core_v1_api


def reset_core_v1_api() -> None:
    global _core_v1_api  # pylint: disable=global-statement
    _core_v1_api = None


class BatchedWarner:
    """Batches messages, emits one summary warning per flush interval window.

    Each call to :meth:`add` appends the message to the per-event file log
    immediately and emits the batched warning once enough time has elapsed
    since the last flush. :meth:`flush` is also intended to be wired to the
    watcher's ``on_stream_end`` so a partial final batch is not lost.
    """

    FLUSH_INTERVAL_SEC = int(os.environ.get("ZETTA_BATCHED_WARN_FLUSH_INTERVAL_SEC", "90"))

    def __init__(self, name: str, log_path: Path | None) -> None:
        self.name = name
        self.log_path = log_path
        self.pending: list[str] = []
        self.last_flush = time.time()

    def add(self, msg: str) -> None:
        self.pending.append(msg)
        append_watcher_log(self.log_path, msg)
        if time.time() - self.last_flush >= self.FLUSH_INTERVAL_SEC:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        n = len(self.pending)
        # One sample per unique bracket-prefix (e.g. "[SIGKILL]" / "[K8s:Evicted]")
        # so distinct event categories are not hidden by repetition.
        seen: set[str] = set()
        sample: list[str] = []
        for msg in self.pending:
            end = msg.find("]")
            category = msg[: end + 1] if end != -1 else msg
            if category in seen:
                continue
            seen.add(category)
            sample.append(msg)
        extra = n - len(sample)
        suffix = f", +{extra} more" if extra > 0 else ""
        body = "\n".join(sample)
        logger.warning(f"{self.name} (total {n} events{suffix}):\n{body}")
        self.pending.clear()
        self.last_flush = time.time()


def resilient_watch(
    list_fn_factory: Callable[[], Callable],
    on_event: Callable,
    *,
    namespace: str,
    description: str,
    stop_event: threading.Event | None = None,
    on_stream_end: Callable[[], None] | None = None,
    on_error: Callable[[], None] | None = None,
    **list_fn_kwargs,
) -> None:
    """Stream K8s objects via watch.Watch, retrying transient errors with backoff.

    ``list_fn_factory`` is called once per stream attempt (inside the retry
    try-block) so a fresh client is picked up after ``on_error()`` invalidates
    a cached one. ``on_error`` runs in the except branch so callers can reset
    cached api clients; the next factory call rebuilds against fresh creds.

    Transient errors are routed through a per-loop :class:`BatchedWarner` so
    a stuck watcher emits one summary per flush interval rather than a fresh
    warning on every retry. The summary flushes immediately when the stream
    recovers.

    Extra ``**list_fn_kwargs`` (e.g. ``field_selector``, ``label_selector``)
    are forwarded to the bound list method for server-side filtering.
    """
    w = watch.Watch()
    backoff = 1.0
    err_batcher = BatchedWarner(f"{description} errors", log_path=None)
    try:
        while not (stop_event and stop_event.is_set()):
            try:
                list_fn = list_fn_factory()
                for event in w.stream(
                    list_fn, namespace=namespace, timeout_seconds=30, **list_fn_kwargs
                ):
                    if stop_event and stop_event.is_set():
                        break
                    on_event(event["object"])
                if on_stream_end is not None:
                    on_stream_end()
                err_batcher.flush()
                backoff = 1.0
            except Exception as err:  # pylint: disable=broad-exception-caught
                status = f":{err.status}" if isinstance(err, ApiException) else ""
                err_batcher.add(f"[{type(err).__name__}{status}] {err}")
                if on_error is not None:
                    on_error()
                if stop_event and stop_event.wait(backoff):
                    break
                backoff = min(backoff * 2, 30.0)
    finally:
        w.stop()
