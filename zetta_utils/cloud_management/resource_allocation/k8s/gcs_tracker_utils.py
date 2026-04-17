"""Sidecar-side helpers for the GCS tracker (Firestore-coupled).

Pure-Python classification + stats live in `gcs_classification` so the
mitmproxy-addon subprocess can use them without pulling in firestore.
"""

from __future__ import annotations

import os
import time

from zetta_utils import log
from zetta_utils.layer.db_layer.backend import DBRowDataT
from zetta_utils.run import RunInfo, update_run_info
from zetta_utils.run.db import POD_STATS_DB

from .gcs_classification import (
    GCSStats,
    classify_gcs_request,
    extract_bucket_from_api_path,
)

__all__ = [
    "GCSStats",
    "classify_gcs_request",
    "extract_bucket_from_api_path",
    "get_pod_name",
    "get_worker_type",
    "read_existing_pod_stats",
    "write_pod_stats",
    "write_region_mismatch",
]

logger = log.get_logger("zetta_utils")


def get_pod_name() -> str:
    """Get pod name from environment."""
    return os.environ.get("POD_NAME", "unknown")


def get_worker_type() -> str:
    """Get worker type from environment, e.g. for grouping pods in head-node aggregation."""
    return os.environ.get("WORKER_TYPE", "unspecified")


def read_existing_pod_stats(run_id: str, pod_name: str) -> dict | None:
    """Read existing pod stats for this pod from its own Firestore document."""
    doc_key = f"{run_id}__{pod_name}"
    try:
        if doc_key not in POD_STATS_DB:
            return None
        return POD_STATS_DB[doc_key]
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def write_pod_stats(run_id: str, pod_name: str, stats_dict: dict) -> None:
    """Write pod stats to a per-pod Firestore document for horizontal scalability."""
    doc_key = f"{run_id}__{pod_name}"
    data: DBRowDataT = {"run_id": run_id, **stats_dict}
    col_keys = tuple(data.keys())
    POD_STATS_DB[(doc_key, col_keys)] = data


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
