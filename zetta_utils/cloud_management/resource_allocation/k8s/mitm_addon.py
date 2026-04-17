# mypy: ignore-errors
# pylint: disable=all

"""mitmproxy addon for GCS request tracking.

Runs in a subprocess started by mitmdump (see gcs_tracker._start_mitmproxy).
The /opt/zetta_utils install is on PYTHONPATH inside the worker image so the
addon can import the shared classifier from `gcs_classification`. We avoid
importing `gcs_tracker_utils` directly because it pulls in firestore, which
the addon does not need.

Uses a queue to decouple the mitmproxy event loop from stats processing,
so response hooks never block even under thousands of concurrent requests.
"""

import atexit
import fcntl
import json
import queue
import sys
import threading
import time

from zetta_utils.cloud_management.resource_allocation.k8s.gcs_classification import (
    GCSStats,
    classify_gcs_request,
    extract_bucket_from_api_path,
)

STATS_FILE = "/tmp/gcs_tracker_stats.json"
WRITE_INTERVAL = 1  # write to file at most every 1 second

_queue = queue.Queue()
_stats = GCSStats()
_last_write_time = 0


def log(msg):
    """Log to stderr for container visibility."""
    sys.stderr.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [GCS-ADDON] {msg}\n")
    sys.stderr.flush()


def write_stats_to_file():
    """Write current stats to file with file locking."""
    global _last_write_time
    try:
        with open(STATS_FILE, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            json.dump(_stats.to_dict(), f)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        _last_write_time = time.time()
    except Exception as e:
        log(f"Failed to write stats: {e}")


def _record(method, path, egress_bytes):
    """Classify and record a single request."""
    if "?" in path:
        path_part, query_part = path.split("?", 1)
    else:
        path_part, query_part = path, ""
    bucket = extract_bucket_from_api_path(path_part)
    if bucket is None:
        if path_part.startswith("/batch/"):
            # Batch API: bundles N sub-ops in one POST. Re-categorised under
            # `_batch` (uncounted vs. Class A/B) in the next step; for now
            # keep the legacy `_unknown` placeholder so we don't lose the
            # request count.
            bucket = "_unknown"
        else:
            # Coverage gap — log the full request so the classifier can be
            # extended. Don't attribute to any billed class.
            log(f"WARNING: unclassified request: {method} {path}")
            _stats.record("_unclassified", None, "unclassified", egress_bytes)
            return
    op_class, operation = classify_gcs_request(method, path_part, query_part)
    _stats.record(bucket, op_class, operation, egress_bytes)


def _process_queue():
    """Consumer thread: drain queue, update stats, write file periodically."""
    global _last_write_time
    while True:
        try:
            method, path, egress_bytes = _queue.get(timeout=1.0)
        except queue.Empty:
            if time.time() - _last_write_time >= WRITE_INTERVAL:
                write_stats_to_file()
            continue

        _record(method, path, egress_bytes)

        # Drain all remaining items without blocking
        while True:
            try:
                method, path, egress_bytes = _queue.get_nowait()
                _record(method, path, egress_bytes)
            except queue.Empty:
                break

        if time.time() - _last_write_time >= WRITE_INTERVAL:
            write_stats_to_file()


def flush_queue():
    """Drain any remaining items and write final stats on shutdown."""
    while True:
        try:
            method, path, egress_bytes = _queue.get_nowait()
            _record(method, path, egress_bytes)
        except queue.Empty:
            break
    write_stats_to_file()


class GCSTracker:
    """mitmproxy addon that tracks GCS requests."""

    def request(self, flow):
        """Called when a request is received."""
        if flow.request.host != "storage.googleapis.com":
            return

        flow.metadata["gcs_method"] = flow.request.method
        flow.metadata["gcs_path"] = flow.request.path

    def response(self, flow):
        """Called when a response is received. Non-blocking — just enqueues."""
        if "gcs_method" not in flow.metadata:
            return

        method = flow.metadata["gcs_method"]
        egress_bytes = 0
        if method == "GET" and flow.response.content:
            egress_bytes = len(flow.response.content)

        _queue.put((method, flow.metadata["gcs_path"], egress_bytes))


log("GCSTracker addon loaded")
atexit.register(flush_queue)
_consumer = threading.Thread(target=_process_queue, daemon=True)
_consumer.start()
addons = [GCSTracker()]
