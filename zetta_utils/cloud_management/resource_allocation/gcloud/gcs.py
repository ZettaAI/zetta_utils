"""
GCS (Google Cloud Storage) utilities.

Provides functions for checking bucket locations and validating
compute/storage region alignment to avoid egress costs.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

from google.cloud import storage

from zetta_utils import log

logger = log.get_logger("zetta_utils")

# GCS region to GCE region mapping
# GCS uses location names that may differ from GCE region names
# Multi-region locations and their corresponding regions
MULTI_REGION_MAPPING = {
    "US": ["us-east1", "us-east4", "us-west1", "us-west2", "us-west3", "us-west4", "us-central1"],
    "EU": ["europe-west1", "europe-west2", "europe-west3", "europe-west4", "europe-north1"],
    "ASIA": ["asia-east1", "asia-east2", "asia-northeast1", "asia-south1", "asia-southeast1"],
}


@lru_cache(maxsize=128)
def get_bucket_location(bucket_name: str) -> str:
    """
    Get the location of a GCS bucket.

    Args:
        bucket_name: Name of the bucket (without gs:// prefix)

    Returns:
        Location string (e.g., 'US', 'us-east1', 'EU', etc.)
    """
    client = storage.Client(project="unused")
    bucket = client.get_bucket(bucket_name)
    return bucket.location


def extract_bucket_from_path(path: str) -> str | None:
    """
    Extract bucket name from a GCS path.

    Args:
        path: GCS path (gs://bucket/path or bucket/path)

    Returns:
        Bucket name or None if not a GCS path
    """
    if not path:
        return None

    if path.startswith("gs://"):
        path = path[5:]
    elif path.startswith("gcs://"):
        path = path[6:]
    elif "://" in path:
        # Not a GCS path
        return None

    # Handle paths like "bucket/path/to/file"
    parts = path.split("/")
    if parts and parts[0]:
        return parts[0]
    return None


def is_region_compatible(bucket_location: str, compute_region: str) -> bool:
    """
    Check if a bucket location is compatible with a compute region.

    Compatible means data transfer won't incur inter-region egress costs.

    Args:
        bucket_location: GCS bucket location (e.g., 'US', 'us-east1')
        compute_region: GCE compute region (e.g., 'us-east1')

    Returns:
        True if compatible (no egress costs), False otherwise
    """
    bucket_location = bucket_location.upper()
    compute_region = compute_region.lower()

    # If bucket is in a multi-region, check if compute region is covered
    if bucket_location in MULTI_REGION_MAPPING:
        return compute_region in MULTI_REGION_MAPPING[bucket_location]

    # If bucket is in a specific region, it must match exactly
    # Bucket locations are uppercase, compute regions are lowercase
    return bucket_location.lower() == compute_region


def check_bucket_region_alignment(
    bucket_names: Iterable[str],
    compute_region: str,
    raise_on_mismatch: bool = False,
) -> dict[str, dict]:
    """
    Check if buckets are in the same region as compute.

    Args:
        bucket_names: Iterable of bucket names to check
        compute_region: The GCE region where compute will run
        raise_on_mismatch: If True, raise an error on region mismatch

    Returns:
        Dict mapping bucket names to their location info:
        {
            "bucket_name": {
                "location": "US",
                "compatible": True,
                "egress_risk": False,
            }
        }

    Raises:
        ValueError: If raise_on_mismatch is True and any bucket is incompatible
    """
    results = {}
    incompatible_buckets = []

    for bucket_name in set(bucket_names):
        try:
            location = get_bucket_location(bucket_name)
            compatible = is_region_compatible(location, compute_region)

            results[bucket_name] = {
                "location": location,
                "compatible": compatible,
                "egress_risk": not compatible,
            }

            if not compatible:
                incompatible_buckets.append((bucket_name, location))

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(f"Could not get location for bucket '{bucket_name}': {e}")
            results[bucket_name] = {
                "location": "UNKNOWN",
                "compatible": None,
                "egress_risk": None,
                "error": str(e),
            }

    if incompatible_buckets:
        msg = (
            f"Potential inter-region egress costs detected!\n"
            f"Compute region: {compute_region}\n"
            f"Incompatible buckets:\n"
        )
        for bucket, location in incompatible_buckets:
            msg += f"  - {bucket}: {location}\n"
        msg += (
            "Consider using buckets in the same region as compute to avoid egress costs.\n"
            "See: https://cloud.google.com/storage/pricing#network-egress"
        )

        if raise_on_mismatch:
            raise ValueError(msg)
        logger.warning(msg)

    return results


def extract_gcs_paths_from_spec(spec: dict | list | tuple | str) -> set[str]:
    """
    Recursively extract all GCS paths from a specification.

    Args:
        spec: A specification dict, list, or string that may contain GCS paths

    Returns:
        Set of bucket names found in the spec
    """
    buckets = set()

    if isinstance(spec, str):
        # Check if it's a GCS path
        bucket = extract_bucket_from_path(spec)
        if bucket:
            buckets.add(bucket)
    elif isinstance(spec, dict):
        for key, value in spec.items():
            # Special handling for kvstore - only extract bucket, don't recurse into other fields
            if key == "kvstore" and isinstance(value, dict):
                if "bucket" in value:
                    buckets.add(value["bucket"])
                continue

            # Common keys that contain GCS paths as strings
            if key in ("path", "bucket", "cloudpath", "base_path", "src", "dst"):
                if isinstance(value, str):
                    bucket = extract_bucket_from_path(value)
                    if bucket:
                        buckets.add(bucket)
                    continue

            # Recurse into nested structures
            if isinstance(value, (dict, list, tuple)):
                buckets.update(extract_gcs_paths_from_spec(value))
    elif isinstance(spec, (list, tuple)):
        for item in spec:
            buckets.update(extract_gcs_paths_from_spec(item))

    return buckets


def validate_region_alignment_for_run(
    spec: dict | list | None,
    compute_region: str,
    raise_on_mismatch: bool = False,
) -> dict[str, dict]:
    """
    Validate that all GCS buckets in a run spec are aligned with compute region.

    Args:
        spec: The run specification containing GCS paths
        compute_region: The GCE region where compute will run
        raise_on_mismatch: If True, raise an error on region mismatch

    Returns:
        Dict with bucket alignment info (see check_bucket_region_alignment)
    """
    if spec is None:
        logger.debug("No spec provided, skipping region alignment check")
        return {}

    buckets = extract_gcs_paths_from_spec(spec)

    if not buckets:
        logger.debug("No GCS buckets found in spec")
        return {}

    logger.info(f"Checking region alignment for {len(buckets)} bucket(s): {buckets}")
    return check_bucket_region_alignment(
        buckets,
        compute_region,
        raise_on_mismatch=raise_on_mismatch,
    )
