"""Tests for GCS utilities."""

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
    """Tests for is_region_compatible function."""

    def test_us_multi_region_compatible(self):
        assert is_region_compatible("US", "us-east1") is True
        assert is_region_compatible("US", "us-west1") is True
        assert is_region_compatible("US", "us-central1") is True

    def test_us_multi_region_incompatible(self):
        assert is_region_compatible("US", "europe-west1") is False
        assert is_region_compatible("US", "asia-east1") is False

    def test_eu_multi_region(self):
        assert is_region_compatible("EU", "europe-west1") is True
        assert is_region_compatible("EU", "europe-north1") is True
        assert is_region_compatible("EU", "us-east1") is False

    def test_asia_multi_region(self):
        assert is_region_compatible("ASIA", "asia-east1") is True
        assert is_region_compatible("ASIA", "asia-southeast1") is True
        assert is_region_compatible("ASIA", "us-east1") is False

    def test_single_region_exact_match(self):
        assert is_region_compatible("us-east1", "us-east1") is True
        assert is_region_compatible("US-EAST1", "us-east1") is True

    def test_single_region_mismatch(self):
        assert is_region_compatible("us-east1", "us-west1") is False
        assert is_region_compatible("europe-west1", "us-east1") is False


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
