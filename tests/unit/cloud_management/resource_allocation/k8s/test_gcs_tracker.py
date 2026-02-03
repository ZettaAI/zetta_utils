"""Tests for GCS request tracker."""

import threading

from zetta_utils.cloud_management.resource_allocation.k8s.gcs_tracker_utils import (
    GCSStats,
    classify_gcs_request,
)


class TestClassifyGcsRequest:
    """Tests for classify_gcs_request function."""

    # Class A operations - DELETE
    def test_delete_is_class_a(self):
        op_class, operation = classify_gcs_request("DELETE", "/storage/v1/b/bucket/o/object")
        assert op_class == "A"
        assert operation == "delete"

    # Class A operations - POST (create/compose/copy/rewrite)
    def test_post_upload_is_class_a(self):
        op_class, operation = classify_gcs_request(
            "POST", "/upload/storage/v1/b/bucket/o?uploadType=media"
        )
        assert op_class == "A"
        assert operation == "insert"

    def test_post_compose_is_class_a(self):
        op_class, operation = classify_gcs_request("POST", "/storage/v1/b/bucket/o/dest/compose")
        assert op_class == "A"
        assert operation == "compose"

    def test_post_copy_is_class_a(self):
        op_class, operation = classify_gcs_request(
            "POST", "/storage/v1/b/src-bucket/o/src-object/copyTo/b/dest-bucket/o/dest-object"
        )
        assert op_class == "A"
        assert operation == "copy"

    def test_post_rewrite_is_class_a(self):
        op_class, operation = classify_gcs_request(
            "POST", "/storage/v1/b/src-bucket/o/src-object/rewriteTo/b/dest-bucket/o/dest-object"
        )
        assert op_class == "A"
        assert operation == "rewrite"

    def test_post_default_is_insert(self):
        op_class, operation = classify_gcs_request("POST", "/storage/v1/b/bucket/o")
        assert op_class == "A"
        assert operation == "insert"

    # Class A operations - PUT (upload/update)
    def test_put_upload_is_class_a(self):
        op_class, operation = classify_gcs_request(
            "PUT", "/upload/storage/v1/b/bucket/o/object?uploadType=resumable"
        )
        assert op_class == "A"
        assert operation == "insert"

    def test_put_update_is_class_a(self):
        op_class, operation = classify_gcs_request("PUT", "/storage/v1/b/bucket/o/object")
        assert op_class == "A"
        assert operation == "update"

    # Class A operations - PATCH
    def test_patch_is_class_a(self):
        op_class, operation = classify_gcs_request("PATCH", "/storage/v1/b/bucket/o/object")
        assert op_class == "A"
        assert operation == "patch"

    # Class A operations - List operations
    def test_get_list_objects_is_class_a(self):
        op_class, operation = classify_gcs_request("GET", "/storage/v1/b/bucket/o?prefix=folder/")
        assert op_class == "A"
        assert operation == "list_objects"

    def test_get_list_with_delimiter_is_class_a(self):
        op_class, operation = classify_gcs_request("GET", "/storage/v1/b/bucket/o?delimiter=/")
        assert op_class == "A"
        assert operation == "list_objects"

    def test_get_list_with_max_results_is_class_a(self):
        op_class, operation = classify_gcs_request("GET", "/storage/v1/b/bucket/o?maxResults=100")
        assert op_class == "A"
        assert operation == "list_objects"

    def test_get_list_with_page_token_is_class_a(self):
        op_class, operation = classify_gcs_request(
            "GET", "/storage/v1/b/bucket/o?pageToken=abc123"
        )
        assert op_class == "A"
        assert operation == "list_objects"

    def test_get_list_objects_endpoint_is_class_a(self):
        op_class, operation = classify_gcs_request("GET", "/storage/v1/b/bucket/o")
        assert op_class == "A"
        assert operation == "list_objects"

    def test_get_list_buckets_is_class_a(self):
        op_class, operation = classify_gcs_request("GET", "/storage/v1/b")
        assert op_class == "A"
        assert operation == "list_buckets"

    # Class B operations - GET object
    def test_get_object_is_class_b(self):
        op_class, operation = classify_gcs_request(
            "GET", "/storage/v1/b/bucket/o/path%2Fto%2Fobject"
        )
        assert op_class == "B"
        assert operation == "get"

    def test_get_object_with_alt_media_is_class_b(self):
        op_class, operation = classify_gcs_request(
            "GET", "/storage/v1/b/bucket/o/object?alt=media"
        )
        assert op_class == "B"
        assert operation == "get"

    # Class B operations - HEAD
    def test_head_is_class_b(self):
        op_class, operation = classify_gcs_request("HEAD", "/storage/v1/b/bucket/o/object")
        assert op_class == "B"
        assert operation == "get_metadata"

    # Unknown operations default to Class B
    def test_unknown_method_is_class_b(self):
        op_class, operation = classify_gcs_request("OPTIONS", "/storage/v1/b/bucket/o/object")
        assert op_class == "B"
        assert operation == "unknown"

    # Case insensitivity
    def test_method_is_case_insensitive(self):
        op_class1, _ = classify_gcs_request("get", "/storage/v1/b/bucket/o/object")
        op_class2, _ = classify_gcs_request("GET", "/storage/v1/b/bucket/o/object")
        op_class3, _ = classify_gcs_request("Get", "/storage/v1/b/bucket/o/object")
        assert op_class1 == op_class2 == op_class3 == "B"


class TestGCSStats:
    """Tests for GCSStats class."""

    def test_initial_state(self):
        stats = GCSStats()
        assert len(stats.buckets) == 0

    def test_record_class_a(self):
        stats = GCSStats()
        stats.record("my-bucket", "A", "insert")

        assert stats.buckets["my-bucket"]["class_a_count"] == 1
        assert stats.buckets["my-bucket"]["class_b_count"] == 0
        assert stats.buckets["my-bucket"]["operations"]["insert"] == 1

    def test_record_class_b(self):
        stats = GCSStats()
        stats.record("my-bucket", "B", "get")

        assert stats.buckets["my-bucket"]["class_a_count"] == 0
        assert stats.buckets["my-bucket"]["class_b_count"] == 1
        assert stats.buckets["my-bucket"]["operations"]["get"] == 1

    def test_record_multiple_operations(self):
        stats = GCSStats()
        stats.record("bucket-a", "A", "insert")
        stats.record("bucket-a", "A", "insert")
        stats.record("bucket-a", "A", "delete")
        stats.record("bucket-b", "B", "get")
        stats.record("bucket-b", "B", "get")

        assert stats.buckets["bucket-a"]["class_a_count"] == 3
        assert stats.buckets["bucket-a"]["class_b_count"] == 0
        assert stats.buckets["bucket-a"]["operations"]["insert"] == 2
        assert stats.buckets["bucket-a"]["operations"]["delete"] == 1

        assert stats.buckets["bucket-b"]["class_a_count"] == 0
        assert stats.buckets["bucket-b"]["class_b_count"] == 2
        assert stats.buckets["bucket-b"]["operations"]["get"] == 2

    def test_to_dict(self):
        stats = GCSStats()
        stats.record("my-bucket", "A", "insert")
        stats.record("my-bucket", "B", "get")

        result = stats.to_dict()

        assert "buckets" in result
        assert result["buckets"]["my-bucket"]["class_a_count"] == 1
        assert result["buckets"]["my-bucket"]["class_b_count"] == 1
        assert result["buckets"]["my-bucket"]["operations"] == {"insert": 1, "get": 1}
        assert "last_updated" in result

    def test_thread_safety(self):
        """Test that concurrent recording doesn't cause race conditions."""
        stats = GCSStats()
        num_threads = 10
        records_per_thread = 100

        def record_operations():
            for _ in range(records_per_thread):
                stats.record("my-bucket", "A", "insert")
                stats.record("my-bucket", "B", "get")

        threads = [threading.Thread(target=record_operations) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected_count = num_threads * records_per_thread
        assert stats.buckets["my-bucket"]["class_a_count"] == expected_count
        assert stats.buckets["my-bucket"]["class_b_count"] == expected_count
