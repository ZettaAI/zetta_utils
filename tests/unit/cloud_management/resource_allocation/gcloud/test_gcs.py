"""Tests for GCS utilities."""

import pytest

from zetta_utils.cloud_management.resource_allocation.gcloud.gcs import (
    extract_bucket_from_path,
    extract_gcs_paths_from_spec,
    is_region_compatible,
)


class TestExtractBucketFromPath:
    """Tests for extract_bucket_from_path function."""

    def test_gs_prefix(self):
        assert extract_bucket_from_path("gs://my-bucket/path/to/file") == "my-bucket"

    def test_gcs_prefix(self):
        assert extract_bucket_from_path("gcs://my-bucket/path/to/file") == "my-bucket"

    def test_no_prefix(self):
        assert extract_bucket_from_path("my-bucket/path/to/file") == "my-bucket"

    def test_bucket_only(self):
        assert extract_bucket_from_path("gs://my-bucket") == "my-bucket"

    def test_non_gcs_path(self):
        assert extract_bucket_from_path("s3://my-bucket/path") is None
        assert extract_bucket_from_path("https://example.com/path") is None

    def test_empty_path(self):
        assert extract_bucket_from_path("") is None


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


class TestExtractGcsPathsFromSpec:
    """Tests for extract_gcs_paths_from_spec function."""

    def test_string_gcs_path(self):
        assert extract_gcs_paths_from_spec("gs://bucket1/path") == {"bucket1"}

    def test_string_non_gcs_path(self):
        assert extract_gcs_paths_from_spec("s3://bucket1/path") == set()

    def test_dict_with_path_key(self):
        spec = {"path": "gs://bucket1/data"}
        assert extract_gcs_paths_from_spec(spec) == {"bucket1"}

    def test_dict_with_bucket_key(self):
        spec = {"bucket": "bucket1"}
        assert extract_gcs_paths_from_spec(spec) == {"bucket1"}

    def test_nested_kvstore(self):
        spec = {
            "kvstore": {
                "driver": "gcs",
                "bucket": "bucket1",
                "path": "data/",
            }
        }
        assert extract_gcs_paths_from_spec(spec) == {"bucket1"}

    def test_list_of_paths(self):
        spec = ["gs://bucket1/a", "gs://bucket2/b", "s3://bucket3/c"]
        assert extract_gcs_paths_from_spec(spec) == {"bucket1", "bucket2"}

    def test_complex_nested_spec(self):
        spec = {
            "src": "gs://source-bucket/input",
            "dst": "gs://dest-bucket/output",
            "layers": [
                {"path": "gs://layer-bucket/layer1"},
                {"cloudpath": "gs://layer-bucket/layer2"},
            ],
            "config": {
                "kvstore": {"bucket": "config-bucket"},
            },
        }
        buckets = extract_gcs_paths_from_spec(spec)
        assert "source-bucket" in buckets
        assert "dest-bucket" in buckets
        assert "layer-bucket" in buckets
        assert "config-bucket" in buckets

    def test_empty_spec(self):
        assert extract_gcs_paths_from_spec({}) == set()
        assert extract_gcs_paths_from_spec([]) == set()
        assert extract_gcs_paths_from_spec("") == set()

    def test_none_values(self):
        spec = {"path": None, "bucket": None}
        assert extract_gcs_paths_from_spec(spec) == set()
