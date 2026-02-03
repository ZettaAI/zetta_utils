# mypy: ignore-errors
# pylint: disable=all

"""mitmproxy addon for GCS request tracking.

This script runs in a subprocess started by mitmdump. It must be self-contained
since it cannot import from the parent process.
"""

import fcntl
import json
import re
import sys
import threading
import time
from collections import defaultdict

STATS_FILE = "/tmp/gcs_tracker_stats.json"


def log(msg):
    """Log to stderr for container visibility."""
    sys.stderr.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [GCS-ADDON] {msg}\n")
    sys.stderr.flush()


_lock = threading.Lock()
_stats = {"buckets": {}}  # Per-bucket stats
_last_write_time = 0
WRITE_INTERVAL = 1  # Write to file at most every 1 second


def _get_bucket_stats(bucket: str) -> dict:
    """Get or create stats dict for a bucket."""
    if bucket not in _stats["buckets"]:
        _stats["buckets"][bucket] = {
            "class_a_count": 0,
            "class_b_count": 0,
            "egress_bytes": 0,
            "operations": defaultdict(int),
        }
    return _stats["buckets"][bucket]


def extract_bucket_from_api_path(path):
    """Extract bucket name from GCS JSON API path.

    Path formats:
    - /storage/v1/b/{bucket}/o/{object}
    - /upload/storage/v1/b/{bucket}/o/...
    - /b/{bucket}/o/...
    """
    match = re.search(r"/b/([^/]+)/", path)
    if match:
        return match.group(1)
    return None


def classify_gcs_request(method, path, query=""):
    """Classify a GCS request as Class A (expensive) or Class B (cheap)."""
    method = method.upper()

    # Check for listing operations (Class A)
    if query:
        if any(p in query for p in ["prefix", "delimiter", "maxResults", "pageToken"]):
            return ("A", "list_objects")

    # DELETE is always Class A
    if method == "DELETE":
        return ("A", "delete")

    # POST operations
    if method == "POST":
        if "/compose" in path:
            return ("A", "compose")
        if "/copyTo/" in path or "/copy" in path:
            return ("A", "copy")
        if "/rewriteTo/" in path or "/rewrite" in path:
            return ("A", "rewrite")
        if "/upload" in path or "uploadType" in query:
            return ("A", "insert")
        return ("A", "insert")

    # PUT operations
    if method == "PUT":
        if "/upload" in path or "uploadType" in query:
            return ("A", "insert")
        return ("A", "update")

    # PATCH is Class A
    if method == "PATCH":
        return ("A", "patch")

    # GET and HEAD operations
    if method in ("GET", "HEAD"):
        if "/o" in path and "/o/" not in path:
            return ("A", "list_objects")
        if "/b" in path and "/b/" not in path:
            return ("A", "list_buckets")
        if method == "HEAD":
            return ("B", "get_metadata")
        return ("B", "get")

    return ("B", "unknown")


def write_stats_to_file():
    """Write current stats to file with locking."""
    global _last_write_time
    with _lock:
        stats_dict = {
            "buckets": {
                bucket: {
                    "class_a_count": data["class_a_count"],
                    "class_b_count": data["class_b_count"],
                    "egress_bytes": data["egress_bytes"],
                    "operations": dict(data["operations"]),
                }
                for bucket, data in _stats["buckets"].items()
            },
            "last_updated": time.time(),
        }
    try:
        with open(STATS_FILE, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            json.dump(stats_dict, f)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        _last_write_time = time.time()
    except Exception as e:
        log(f"Failed to write stats: {e}")


def record_request(method, path, egress_bytes=0):
    """Record a GCS request."""
    global _last_write_time
    if "?" in path:
        path_part, query_part = path.split("?", 1)
    else:
        path_part, query_part = path, ""

    op_class, operation = classify_gcs_request(method, path_part, query_part)
    bucket = extract_bucket_from_api_path(path_part) or "_unknown"

    with _lock:
        bucket_stats = _get_bucket_stats(bucket)
        if op_class == "A":
            bucket_stats["class_a_count"] += 1
        else:
            bucket_stats["class_b_count"] += 1
        bucket_stats["operations"][operation] += 1
        bucket_stats["egress_bytes"] += egress_bytes

    # Write stats to file periodically
    if time.time() - _last_write_time >= WRITE_INTERVAL:
        write_stats_to_file()


class GCSTracker:
    """mitmproxy addon that tracks GCS requests."""

    def request(self, flow):
        """Called when a request is received."""
        if "storage.googleapis.com" not in flow.request.host:
            return

        flow.metadata["gcs_method"] = flow.request.method
        flow.metadata["gcs_path"] = flow.request.path
        # log(f"GCS request tracked: {flow.request.method} {flow.request.path[:100]}")

    def response(self, flow):
        """Called when a response is received."""
        if "gcs_method" not in flow.metadata:
            return

        method = flow.metadata["gcs_method"]
        # Only count egress bytes for GET requests (actual downloads)
        egress_bytes = 0
        if method == "GET" and flow.response.content:
            egress_bytes = len(flow.response.content)

        record_request(method, flow.metadata["gcs_path"], egress_bytes=egress_bytes)
        # log(f"GCS response recorded: {method} egress={egress_bytes}")


log("GCSTracker addon loaded")
addons = [GCSTracker()]
