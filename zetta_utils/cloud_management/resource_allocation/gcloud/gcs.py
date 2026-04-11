"""
GCS (Google Cloud Storage) utilities.

Provides functions for checking bucket locations and validating
compute/storage region alignment to avoid egress costs.
"""

from __future__ import annotations

from functools import lru_cache

from google.cloud import storage  # type: ignore[attr-defined]

from zetta_utils import log

logger = log.get_logger("zetta_utils")

# Goal: detect cross-region egress no matter what. The function returns True
# only when we have explicit positive evidence the bucket and compute share a
# region; everything else (including unknown new GCE regions) is flagged as
# potential egress. False positives are noise; false negatives are silent
# money loss, so we err on the side of false positives.

# Known-good positive list of regions in each multi-region's replica set, as
# of April 2026. GCP does not publish an authoritative per-multi-region
# constituent list, so this is maintained manually. When a brand-new GCE
# region is added by GCP, it will be flagged as potential egress until it is
# explicitly added here.
MULTI_REGION_REGIONS: dict[str, frozenset[str]] = {
    "US": frozenset(
        {
            "us-central1",
            "us-east1",
            "us-east4",
            "us-east5",
            "us-south1",
            "us-west1",
            "us-west2",
            "us-west3",
            "us-west4",
        }
    ),
    "EU": frozenset(
        {
            "europe-central2",
            "europe-north1",
            "europe-north2",
            "europe-southwest1",
            "europe-west1",
            "europe-west3",
            "europe-west4",
            "europe-west8",
            "europe-west9",
            "europe-west10",
            "europe-west12",
        }
    ),
    "ASIA": frozenset(
        {
            "asia-east1",
            "asia-northeast1",
            "asia-northeast2",
            "asia-northeast3",
            "asia-south1",
            "asia-south2",
            "asia-southeast1",
            "asia-southeast3",
        }
    ),
}

# Documented exclusions — regions geographically inside a multi-region's area
# that are explicitly NOT part of its replica set per the official GCP docs.
# Not used by `is_region_compatible` directly (the positive list above already
# omits them); kept here so the omissions in MULTI_REGION_REGIONS are
# intentional, documented, and not silently "fixed" by a future maintainer.
# https://cloud.google.com/storage/docs/locations
MULTI_REGION_EXCLUSIONS: dict[str, frozenset[str]] = {
    "US": frozenset(),
    "EU": frozenset({"europe-west2", "europe-west6"}),  # London, Zurich
    "ASIA": frozenset({"asia-east2", "asia-southeast2"}),  # Hong Kong, Jakarta
}

# Predefined dual-regions and their constituent regions. The GCS API only
# returns the location code (e.g. "NAM4") for these — bucket.data_locations
# is None — so the mapping must live client-side.
# https://cloud.google.com/storage/docs/locations
PREDEFINED_DUAL_REGIONS: dict[str, frozenset[str]] = {
    "NAM4": frozenset({"us-central1", "us-east1"}),
    "EUR4": frozenset({"europe-north1", "europe-west4"}),
    "EUR5": frozenset({"europe-west1", "europe-west2"}),
    "EUR7": frozenset({"europe-west2", "europe-west3"}),
    "EUR8": frozenset({"europe-west3", "europe-west6"}),
    "ASIA1": frozenset({"asia-northeast1", "asia-northeast2"}),
}


@lru_cache(maxsize=128)
def get_bucket_location_info(bucket_name: str) -> tuple[str, tuple[str, ...] | None]:
    """
    Get the location code and constituent data locations of a GCS bucket.

    Returns:
        A tuple of (location_code, data_locations).
        - location_code: e.g., 'US', 'us-east1', 'NAM4'.
        - data_locations: tuple of constituent regions for configurable
          dual-region buckets, or None for multi-regions, predefined
          dual-regions, and single-region buckets (the API only exposes
          per-bucket constituent regions for configurable dual-regions).
    """
    client = storage.Client(project="unused")
    bucket = client.get_bucket(bucket_name)
    data_locs = bucket.data_locations
    return bucket.location, tuple(data_locs) if data_locs else None


def get_bucket_location(bucket_name: str) -> str:
    """Get the location code of a GCS bucket. Prefer `get_bucket_location_info`."""
    return get_bucket_location_info(bucket_name)[0]


def is_region_compatible(
    bucket_location: str,
    compute_region: str,
    bucket_data_locations: tuple[str, ...] | None = None,
) -> bool:
    """Check if a bucket location is compatible with a compute region.

    Compatible == data transfer between this bucket and a VM in `compute_region`
    will not incur cross-region egress charges.

    Strategy: only return True with explicit positive evidence. Anything we
    cannot definitively classify as same-region is treated as cross-region so
    that egress is never silently missed (false positives over false negatives).

    Cases, in order:

    1. **Configurable dual-region** — `bucket_data_locations` is the API-
       reported tuple of constituent regions for user-defined dual-regions.
       When non-None, this is the authoritative answer; we ignore the static
       maps and check membership directly.

    2. **Predefined dual-region** (NAM4, EUR4, ...) — the API only returns the
       location code; the constituent pair is hard-coded in
       `PREDEFINED_DUAL_REGIONS`. Reads from a VM in either constituent region
       are routed locally and incur no egress.

    3. **Multi-region** (US, EU, ASIA) — checked against the explicit positive
       list `MULTI_REGION_REGIONS`. Brand-new GCE regions GCP adds will be
       flagged as incompatible until they are explicitly added to the list,
       which is the safe failure mode. The companion `MULTI_REGION_EXCLUSIONS`
       documents regions known to be outside the replica set
       (London/Zurich for EU, Hong Kong/Jakarta for ASIA) so the omissions in
       the positive list are intentional and not silently re-added later.

    4. **Single region** — exact case-insensitive match.
    """
    compute_region = compute_region.lower()

    # Case 1: configurable dual-region — API tells us the constituent regions.
    if bucket_data_locations:
        return compute_region in {r.lower() for r in bucket_data_locations}

    bucket_location = bucket_location.upper()

    # Case 2: predefined dual-region.
    if bucket_location in PREDEFINED_DUAL_REGIONS:
        return compute_region in PREDEFINED_DUAL_REGIONS[bucket_location]

    # Case 3: multi-region — explicit positive list, conservative.
    if bucket_location in MULTI_REGION_REGIONS:
        return compute_region in MULTI_REGION_REGIONS[bucket_location]

    # Case 4: single-region exact match.
    return bucket_location.lower() == compute_region
