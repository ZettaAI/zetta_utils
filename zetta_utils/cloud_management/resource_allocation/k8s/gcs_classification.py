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
from typing import Callable, Literal


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
        self,
        bucket: str,
        op_class: Literal["A", "B"] | None,
        operation: str,
        egress_bytes: int = 0,
    ):
        """Record one operation against `bucket`.

        `op_class=None` means "do not attribute to a billed class" — the
        operation count and egress still increment, but neither Class A
        nor Class B totals do. Use for `_unclassified` (classifier
        coverage gap) and `_batch` (sub-ops not yet parsed) entries.
        """
        with self.lock:
            bucket_stats = self._get_bucket_stats(bucket)
            if op_class == "A":
                bucket_stats["class_a_count"] += 1
            elif op_class == "B":
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


# Path prefixes that identify GCS JSON-style APIs. Used so the classifier
# only applies JSON-API substring heuristics on paths that actually have
# JSON-API semantics. Without this gate, the `/o` and `/b` substring checks
# in _classify_get_head misfire on XML-API object paths like
# `/{bucket}/some/object` (e.g. `/o` matches `.../object`) and flip Class B
# reads to Class A "list_objects" — the source of the inflated cost
# previously seen against the `_unknown` bucket.
_JSON_API_PREFIXES = (
    "/storage/",
    "/upload/storage/",
    "/download/storage/",
    "/resumable/",
)

# Query params that indicate a list-objects operation. JSON API uses
# prefix/delimiter/maxResults/pageToken; XML (S3-compatible) API uses
# list-type/marker/max-keys.
_LIST_QUERY_PARAMS = (
    "prefix",
    "delimiter",
    "maxResults",
    "pageToken",
    "list-type",
    "marker",
    "max-keys",
)

# First path segments that are NOT XML-API buckets — internal/GAE paths or
# JSON-API prefixes that should have matched the JSON branch already.
_NON_BUCKET_FIRST_SEGMENTS = frozenset(
    {"_ah", "storage", "upload", "download", "resumable", "batch"}
)


def _is_json_api_path(path: str) -> bool:
    return path.startswith(_JSON_API_PREFIXES)


def extract_bucket_from_api_path(path: str) -> str | None:
    """Extract bucket name from a GCS request path.

    GCS exposes several URL shapes on storage.googleapis.com:

    - **JSON API** (`/storage/v1/b/{bucket}[/...]`, with optional
      `/upload/`, `/download/`, `/resumable/` prefix). Bucket follows `/b/`.
    - **Batch API** (`/batch/storage/v1`). No single bucket — caller
      attributes to a separate `_batch` key.
    - **XML API** (`/{bucket}[/{object}]`). First path segment is the
      bucket. This is what cloudvolume / tensorstore use for object I/O,
      and is the bulk of traffic — previously missed by the regex below.
    """
    # JSON API. `[^/?]+` so a bucket-level path without trailing slash
    # (e.g. `/storage/v1/b/{bucket}?fields=...`) still matches.
    m = re.match(r"/(?:upload/|download/|resumable/)?storage/v\d+/b/([^/?]+)", path)
    if m:
        return m.group(1)

    # Batch API: caller decides what to do (track under `_batch`, etc.).
    if path.startswith("/batch/"):
        return None

    # XML API: first path segment is the bucket. Reject known internal /
    # JSON-API prefixes (defensive — those should have matched above).
    m = re.match(r"/([^/?]+)", path)
    if m and m.group(1) not in _NON_BUCKET_FIRST_SEGMENTS:
        return m.group(1)
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
    """Classify PUT requests (all Class A).

    JSON API distinguishes upload (insert) from metadata update via the
    `/upload/` prefix or `uploadType` query param. XML API `PUT
    /{bucket}/{object}` is always an object upload (insert).
    """
    if _is_json_api_path(path):
        if "/upload" in path or "uploadType" in query:
            return ("A", "insert")
        return ("A", "update")
    return ("A", "insert")


def _classify_get_head(path: str, method: str) -> tuple[Literal["A", "B"], str]:
    """Classify GET/HEAD requests.

    JSON API distinguishes list vs. get by URL shape: `.../{bucket}/o`
    (without trailing object) is `list_objects`; `.../{bucket}/o/{name}`
    is `get`. These substring checks are gated on the path actually
    being a JSON-API path — otherwise they misfire on XML-API object
    names that happen to contain `/o` or `/b` substrings.
    """
    if _is_json_api_path(path):
        if "/o" in path and "/o/" not in path:
            return ("A", "list_objects")
        if "/b" in path and "/b/" not in path:
            return ("A", "list_buckets")
    if method == "HEAD":
        return ("B", "get_metadata")
    return ("B", "get")


# Synthetic bucket keys for requests that don't map to a single billable
# bucket. Surfaced in the per-pod stats payload so they're visible in cost
# breakdowns alongside real buckets.
BATCH_BUCKET = "_batch"
UNCLASSIFIED_BUCKET = "_unclassified"


def route_request(
    stats: GCSStats,
    method: str,
    path: str,
    egress_bytes: int = 0,
    on_unclassified: Callable[[str, str], None] | None = None,
) -> None:
    """Classify a single GCS request and record it into `stats`.

    Bucket categories:
    - real bucket name → classified per `classify_gcs_request` (Class A or B).
    - `_batch` → `/batch/storage/v1` calls. One HTTP request can carry N
      sub-ops, so we count the request but do not attribute to a billed
      class until the multipart body is parsed (deferred).
    - `_unclassified` → `extract_bucket_from_api_path` returned None for a
      non-batch path. Coverage gap. `on_unclassified(method, path)` is
      called (if provided) so callers can log loudly. Operation/egress
      still recorded; billed totals are not.
    """
    path_part, _, query_part = path.partition("?")
    bucket = extract_bucket_from_api_path(path_part)
    if bucket is not None:
        op_class, operation = classify_gcs_request(method, path_part, query_part)
        stats.record(bucket, op_class, operation, egress_bytes)
        return
    if path_part.startswith("/batch/"):
        stats.record(BATCH_BUCKET, None, "batch", egress_bytes)
        return
    if on_unclassified is not None:
        on_unclassified(method, path)
    stats.record(UNCLASSIFIED_BUCKET, None, "unclassified", egress_bytes)


def classify_gcs_request(  # pylint: disable=too-many-return-statements
    method: str, path: str, query: str = ""
) -> tuple[Literal["A", "B"], str]:
    """Classify a GCS request as Class A (expensive) or Class B (cheap).

    GCP pricing classes:
    - Class A (expensive): insert, delete, copy, compose, rewrite, list operations
    - Class B (cheap): get, get_metadata
    """
    method = method.upper()

    if query and any(p in query for p in _LIST_QUERY_PARAMS):
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
