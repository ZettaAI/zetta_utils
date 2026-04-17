"""Pure-Python GCS request classification + per-bucket stats.

Kept dependency-free (stdlib only) so the mitmproxy addon subprocess can
import this without pulling in firestore / db_layer / run modules. Both
`gcs_tracker_utils` (sidecar process) and `mitm_addon` (mitmdump subprocess)
share this implementation — there used to be two near-identical copies and
they drifted, producing inflated Class A counts against the `_unknown`
bucket.
"""

from __future__ import annotations

import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class GCSStats:
    """Thread-safe GCS operation statistics per bucket."""

    buckets: dict = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def _get_bucket_stats(self, bucket: str) -> dict:
        if bucket not in self.buckets:
            self.buckets[bucket] = {
                "class_a_count": 0,
                "class_b_count": 0,
                "egress_bytes": 0,
                "operations": defaultdict(int),
            }
        return self.buckets[bucket]

    def record(
        self, bucket: str, op_class: Literal["A", "B"], operation: str, egress_bytes: int = 0
    ):
        with self.lock:
            bucket_stats = self._get_bucket_stats(bucket)
            if op_class == "A":
                bucket_stats["class_a_count"] += 1
            else:
                bucket_stats["class_b_count"] += 1
            bucket_stats["operations"][operation] += 1
            bucket_stats["egress_bytes"] += egress_bytes

    def to_dict(self) -> dict:
        with self.lock:
            return {
                "buckets": {
                    bucket: {
                        "class_a_count": data["class_a_count"],
                        "class_b_count": data["class_b_count"],
                        "egress_bytes": data["egress_bytes"],
                        "operations": dict(data["operations"]),
                    }
                    for bucket, data in self.buckets.items()
                },
                "last_updated": time.time(),
            }

    def load_from_dict(self, data: dict):
        with self.lock:
            self.buckets.clear()
            for bucket, bucket_data in data.get("buckets", {}).items():
                self.buckets[bucket] = {
                    "class_a_count": bucket_data.get("class_a_count", 0),
                    "class_b_count": bucket_data.get("class_b_count", 0),
                    "egress_bytes": bucket_data.get("egress_bytes", 0),
                    "operations": defaultdict(int, bucket_data.get("operations", {})),
                }


def extract_bucket_from_api_path(path: str) -> str | None:
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


def _classify_post(path: str) -> tuple[Literal["A", "B"], str]:
    """Classify POST requests (all Class A)."""
    if "/compose" in path:
        return ("A", "compose")
    if "/copyTo/" in path or "/copy" in path:
        return ("A", "copy")
    if "/rewriteTo/" in path or "/rewrite" in path:
        return ("A", "rewrite")
    return ("A", "insert")


def _classify_put(path: str, query: str) -> tuple[Literal["A", "B"], str]:
    """Classify PUT requests (all Class A)."""
    if "/upload" in path or "uploadType" in query:
        return ("A", "insert")
    return ("A", "update")


def _classify_get_head(path: str, method: str) -> tuple[Literal["A", "B"], str]:
    """Classify GET/HEAD requests."""
    if "/o" in path and "/o/" not in path:
        return ("A", "list_objects")
    if "/b" in path and "/b/" not in path:
        return ("A", "list_buckets")
    if method == "HEAD":
        return ("B", "get_metadata")
    return ("B", "get")


def classify_gcs_request(  # pylint: disable=too-many-return-statements
    method: str, path: str, query: str = ""
) -> tuple[Literal["A", "B"], str]:
    """Classify a GCS request as Class A (expensive) or Class B (cheap).

    GCP pricing classes:
    - Class A (expensive): insert, delete, copy, compose, rewrite, list operations
    - Class B (cheap): get, get_metadata
    """
    method = method.upper()

    if query and any(p in query for p in ["prefix", "delimiter", "maxResults", "pageToken"]):
        return ("A", "list_objects")

    if method == "DELETE":
        return ("A", "delete")
    if method == "POST":
        return _classify_post(path)
    if method == "PUT":
        return _classify_put(path, query)
    if method == "PATCH":
        return ("A", "patch")
    if method in ("GET", "HEAD"):
        return _classify_get_head(path, method)

    return ("B", "unknown")
