"""Shared utilities for GCS tracking."""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

from zetta_utils import log
from zetta_utils.layer.db_layer.backend import DBRowDataT
from zetta_utils.run import RunInfo, update_run_info
from zetta_utils.run.db import RUN_DB

logger = log.get_logger("zetta_utils")


@dataclass
class GCSStats:
    """Thread-safe GCS operation statistics per bucket."""

    buckets: dict = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def _get_bucket_stats(self, bucket: str) -> dict:
        """Get or create stats dict for a bucket."""
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
    """
    Classify a GCS request as Class A (expensive) or Class B (cheap).

    GCP pricing classes:
    - Class A (expensive): insert, delete, copy, compose, rewrite, list operations
    - Class B (cheap): get, get_metadata
    """
    method = method.upper()

    # Check for listing operations via query params (Class A)
    if query and any(p in query for p in ["prefix", "delimiter", "maxResults", "pageToken"]):
        return ("A", "list_objects")

    # Dispatch by method
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


def get_pod_name() -> str:
    """Get pod name from environment."""
    return os.environ.get("POD_NAME", "unknown")


def read_existing_stats(run_id: str, pod_name: str, field_name: str) -> dict | None:
    """Read existing stats for this pod from Firestore."""
    try:
        if run_id not in RUN_DB:
            return None
        full_doc = RUN_DB[run_id]
        stats_map = full_doc.get(field_name)
        if isinstance(stats_map, dict) and pod_name in stats_map:
            return stats_map[pod_name]
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    return None


def write_stats(run_id: str, pod_name: str, field_name: str, stats_dict: dict) -> None:
    """Write stats to Firestore using dot notation for per-pod storage."""
    field_key = f"{field_name}.{pod_name}"
    info: DBRowDataT = {field_key: stats_dict}
    update_run_info(run_id, info)


def write_region_mismatch(run_id: str, bucket: str, location: str, compute_region: str) -> None:
    """Write region mismatch error to run_db for main container to detect."""
    error: dict = {
        "bucket": bucket,
        "bucket_location": location,
        "compute_region": compute_region,
        "message": f"Bucket '{bucket}' in {location} incompatible with compute {compute_region}",
        "timestamp": time.time(),
    }
    info: DBRowDataT = {RunInfo.REGION_MISMATCH.value: error}
    update_run_info(run_id, info)
    logger.error(f"Region mismatch written to run_db: {error['message']}")
