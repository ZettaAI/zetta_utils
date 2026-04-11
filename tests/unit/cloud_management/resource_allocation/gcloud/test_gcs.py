"""Tests for GCS utilities."""

import pytest

from zetta_utils.cloud_management.resource_allocation.gcloud.gcs import (
    is_region_compatible,
)


class TestIsRegionCompatible:
    """Tests for is_region_compatible function.

    The function is set membership over data tables, so the cases below cover:
    (a) every code path in the function, (b) one positive case per multi-region
    group, (c) the documented exclusion behavior, (d) the conservative
    fall-through for unknown regions, (e) one positive case per predefined
    dual-region (data sanity), and (f) case-insensitivity + the
    `data_locations` override path.
    """

    @pytest.mark.parametrize(
        "bucket,compute,expected",
        [
            # One positive + one outside-the-area negative per multi-region group.
            ("US", "us-east1", True),
            ("US", "europe-west1", False),
            ("EU", "europe-north1", True),
            ("EU", "us-east1", False),
            ("ASIA", "asia-east1", True),
            ("ASIA", "us-east1", False),
            # Documented exclusions (London/Zurich for EU, HK/Jakarta for ASIA):
            # must be flagged as cross-region.
            ("EU", "europe-west2", False),
            ("EU", "europe-west6", False),
            ("ASIA", "asia-east2", False),
            ("ASIA", "asia-southeast2", False),
            # Conservative branch: an unknown new region inside a multi-region's
            # area is flagged rather than silently treated as compatible.
            ("US", "us-fictional99", False),
        ],
    )
    def test_multi_region(self, bucket, compute, expected):
        assert is_region_compatible(bucket, compute) is expected

    @pytest.mark.parametrize(
        "bucket,compute,expected",
        [
            # One positive case per predefined dual-region (data sanity check).
            ("NAM4", "us-east1", True),
            ("EUR4", "europe-west4", True),
            ("EUR5", "europe-west1", True),
            ("EUR7", "europe-west3", True),
            ("EUR8", "europe-west6", True),
            ("ASIA1", "asia-northeast1", True),
            # One negative case + case-insensitivity exercise the lookup logic.
            ("NAM4", "us-west1", False),
            ("nam4", "us-east1", True),
        ],
    )
    def test_predefined_dual_region(self, bucket, compute, expected):
        assert is_region_compatible(bucket, compute) is expected

    @pytest.mark.parametrize(
        "bucket,compute,expected",
        [
            ("us-east1", "us-east1", True),
            ("US-EAST1", "us-east1", True),  # case-insensitive
            ("us-east1", "us-west1", False),
        ],
    )
    def test_single_region(self, bucket, compute, expected):
        assert is_region_compatible(bucket, compute) is expected

    @pytest.mark.parametrize(
        "compute,data_locations,expected",
        [
            # data_locations takes precedence — bucket_location string is ignored.
            ("us-east1", ("us-east1", "us-west1"), True),
            ("us-central1", ("us-east1", "us-west1"), False),
        ],
    )
    def test_configurable_dual_region_uses_data_locations(self, compute, data_locations, expected):
        assert is_region_compatible("US", compute, data_locations) is expected
