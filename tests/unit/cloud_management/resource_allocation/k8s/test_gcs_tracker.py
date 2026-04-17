"""Tests for GCS request tracker."""

import json
import threading

from zetta_utils.cloud_management.resource_allocation.k8s import gcs_tracker
from zetta_utils.cloud_management.resource_allocation.k8s.gcs_classification import (
    BATCH_BUCKET,
    UNCLASSIFIED_BUCKET,
    route_request,
)
from zetta_utils.cloud_management.resource_allocation.k8s.gcs_tracker import (
    StatsCollector,
    _collect_all,
    _collect_resource_stats,
    _collect_semaphore_stats,
    _make_gcs_collector,
)
from zetta_utils.cloud_management.resource_allocation.k8s.gcs_tracker_utils import (
    GCSStats,
    classify_gcs_request,
    extract_bucket_from_api_path,
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


class TestClassifyGcsRequestXmlApi:
    """Classifier behaviour on XML-API paths (`/{bucket}/{object}`).

    cloudvolume and tensorstore use the XML API for object I/O, which is
    the bulk of GCS traffic. Previously the JSON-API substring heuristics
    (`/o`, `/b`) misfired on XML paths and inflated Class A counts.
    """

    def test_get_object_is_class_b(self):
        op_class, operation = classify_gcs_request("GET", "/my-bucket/path/to/object.bin")
        assert op_class == "B"
        assert operation == "get"

    def test_get_object_with_o_substring_is_class_b(self):
        # Regression: previously misclassified as Class A list_objects because
        # the JSON-API `/o` heuristic matched the `.../object` substring.
        op_class, operation = classify_gcs_request("GET", "/my-bucket/output/file.txt")
        assert op_class == "B"
        assert operation == "get"

    def test_head_object_is_class_b(self):
        op_class, operation = classify_gcs_request("HEAD", "/my-bucket/object")
        assert op_class == "B"
        assert operation == "get_metadata"

    def test_put_object_is_class_a_insert(self):
        # XML PUT uploads an object — Class A insert, not "update".
        op_class, operation = classify_gcs_request("PUT", "/my-bucket/path/to/object")
        assert op_class == "A"
        assert operation == "insert"

    def test_delete_object_is_class_a(self):
        op_class, operation = classify_gcs_request("DELETE", "/my-bucket/object")
        assert op_class == "A"
        assert operation == "delete"

    def test_list_via_prefix_is_class_a(self):
        op_class, operation = classify_gcs_request("GET", "/my-bucket/", "prefix=folder/")
        assert op_class == "A"
        assert operation == "list_objects"

    def test_list_via_marker_is_class_a(self):
        op_class, operation = classify_gcs_request("GET", "/my-bucket/", "marker=abc")
        assert op_class == "A"
        assert operation == "list_objects"


class TestExtractBucketFromApiPath:
    """Bucket extraction for JSON, XML, and batch API paths."""

    def test_json_object_path(self):
        assert extract_bucket_from_api_path("/storage/v1/b/my-bucket/o/object") == "my-bucket"

    def test_json_upload_path(self):
        assert (
            extract_bucket_from_api_path("/upload/storage/v1/b/my-bucket/o?uploadType=media")
            == "my-bucket"
        )

    def test_json_download_path(self):
        assert extract_bucket_from_api_path("/download/storage/v1/b/my-bucket/o/x") == "my-bucket"

    def test_json_resumable_path(self):
        assert extract_bucket_from_api_path("/resumable/storage/v1/b/my-bucket/o/x") == "my-bucket"

    def test_json_bucket_level_no_trailing_slash(self):
        # Edge case the old regex missed (required trailing /).
        assert extract_bucket_from_api_path("/storage/v1/b/my-bucket?fields=name") == "my-bucket"

    def test_xml_object_path(self):
        assert extract_bucket_from_api_path("/my-bucket/path/to/object") == "my-bucket"

    def test_xml_object_root(self):
        assert extract_bucket_from_api_path("/my-bucket/object") == "my-bucket"

    def test_xml_bucket_only(self):
        assert extract_bucket_from_api_path("/my-bucket?prefix=foo") == "my-bucket"

    def test_batch_api_returns_none(self):
        # Caller attributes batch ops separately (Part 9).
        assert extract_bucket_from_api_path("/batch/storage/v1") is None

    def test_internal_path_returns_none(self):
        assert extract_bucket_from_api_path("/_ah/health") is None


class TestRouteRequest:
    """Routing of a single request into the right bucket category."""

    def test_real_bucket_classified_as_a(self):
        stats = GCSStats()
        route_request(stats, "PUT", "/my-bucket/path/to/object")
        assert stats.buckets["my-bucket"]["class_a_count"] == 1
        assert stats.buckets["my-bucket"]["operations"]["insert"] == 1

    def test_real_bucket_classified_as_b(self):
        stats = GCSStats()
        route_request(stats, "GET", "/my-bucket/object", egress_bytes=100)
        assert stats.buckets["my-bucket"]["class_b_count"] == 1
        assert stats.buckets["my-bucket"]["egress_bytes"] == 100

    def test_batch_path_routes_to_batch_bucket_uncounted(self):
        stats = GCSStats()
        route_request(stats, "POST", "/batch/storage/v1")
        assert BATCH_BUCKET in stats.buckets
        assert stats.buckets[BATCH_BUCKET]["operations"]["batch"] == 1
        # Don't attribute to any billed class until sub-ops are parsed.
        assert stats.buckets[BATCH_BUCKET]["class_a_count"] == 0
        assert stats.buckets[BATCH_BUCKET]["class_b_count"] == 0

    def test_unrecognised_path_routes_to_unclassified_and_calls_callback(self):
        stats = GCSStats()
        seen: list[tuple[str, str]] = []
        route_request(
            stats,
            "GET",
            "/_ah/health",
            on_unclassified=lambda m, p: seen.append((m, p)),
        )
        assert UNCLASSIFIED_BUCKET in stats.buckets
        assert stats.buckets[UNCLASSIFIED_BUCKET]["class_a_count"] == 0
        assert stats.buckets[UNCLASSIFIED_BUCKET]["class_b_count"] == 0
        assert seen == [("GET", "/_ah/health")]

    def test_query_params_split_correctly(self):
        # `?prefix=…` should be parsed as a list query, not a path component.
        stats = GCSStats()
        route_request(stats, "GET", "/my-bucket/?prefix=folder")
        assert stats.buckets["my-bucket"]["operations"]["list_objects"] == 1
        assert stats.buckets["my-bucket"]["class_a_count"] == 1


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

    def test_record_uncounted(self):
        # op_class=None — operation count + egress increment, but neither
        # class total does. Used for _unclassified / _batch buckets that
        # aren't (yet) attributed to a billed class.
        stats = GCSStats()
        stats.record("_unclassified", None, "unclassified", egress_bytes=42)

        assert stats.buckets["_unclassified"]["class_a_count"] == 0
        assert stats.buckets["_unclassified"]["class_b_count"] == 0
        assert stats.buckets["_unclassified"]["operations"]["unclassified"] == 1
        assert stats.buckets["_unclassified"]["egress_bytes"] == 42

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


class TestCollectAll:
    """Tests for _collect_all collector iteration and payload assembly."""

    def test_empty_collectors_returns_empty_payload(self):
        assert not _collect_all([])

    def test_gcs_collector_merges_flat(self):
        gcs_result = {"buckets": {"b1": {"class_a_count": 5}}, "last_updated": 123.0}
        collector = StatsCollector(name="gcs_stats", collect=lambda: gcs_result)

        payload = _collect_all([collector])

        assert payload == gcs_result

    def test_non_gcs_collector_nests_under_name(self):
        collector = StatsCollector(name="resource_stats", collect=lambda: {"cpu": 50.0})

        payload = _collect_all([collector])

        assert payload == {"resource_stats": {"cpu": 50.0}}

    def test_collector_returning_none_is_omitted(self):
        gcs = StatsCollector(name="gcs_stats", collect=lambda: {"buckets": {}})
        absent = StatsCollector(name="resource_stats", collect=lambda: None)

        payload = _collect_all([gcs, absent])

        assert "resource_stats" not in payload
        assert payload == {"buckets": {}}

    def test_collector_exception_is_caught_and_others_run(self):
        def failing():
            raise RuntimeError("boom")

        ok = StatsCollector(name="resource_stats", collect=lambda: {"cpu": 1.0})
        bad = StatsCollector(name="semaphore_stats", collect=failing)

        payload = _collect_all([bad, ok])

        assert payload == {"resource_stats": {"cpu": 1.0}}

    def test_multiple_collectors_combine(self):
        gcs = StatsCollector(name="gcs_stats", collect=lambda: {"buckets": {"b1": {}}})
        res = StatsCollector(name="resource_stats", collect=lambda: {"cpu": 10.0})
        sem = StatsCollector(name="semaphore_stats", collect=lambda: {"read": {"wait": 0.5}})

        payload = _collect_all([gcs, res, sem])

        assert payload == {
            "buckets": {"b1": {}},
            "resource_stats": {"cpu": 10.0},
            "semaphore_stats": {"read": {"wait": 0.5}},
        }


class TestMakeGcsCollector:
    """Tests for the GCS collector factory."""

    def _write_stats_file(self, tmp_path, contents):
        path = tmp_path / "gcs_tracker_stats.json"
        path.write_text(json.dumps(contents))
        return str(path)

    def test_collector_has_gcs_stats_name(self, tmp_path, mocker):
        mocker.patch.object(
            gcs_tracker, "STATS_FILE", self._write_stats_file(tmp_path, {"buckets": {}})
        )
        collector = _make_gcs_collector(run_id="run-1", compute_region=None)

        assert collector.name == "gcs_stats"

    def test_collect_reads_file(self, tmp_path, mocker):
        contents = {"buckets": {"my-bucket": {"class_a_count": 3}}, "last_updated": 1.0}
        mocker.patch.object(gcs_tracker, "STATS_FILE", self._write_stats_file(tmp_path, contents))
        collector = _make_gcs_collector(run_id="run-1", compute_region=None)

        result = collector.collect()

        assert result == contents

    def test_collect_no_region_check_when_compute_region_none(self, tmp_path, mocker):
        contents = {"buckets": {"my-bucket": {"class_a_count": 1}}, "last_updated": 0.0}
        mocker.patch.object(gcs_tracker, "STATS_FILE", self._write_stats_file(tmp_path, contents))
        check_spy = mocker.patch.object(gcs_tracker, "_check_and_cache_bucket_region")
        collector = _make_gcs_collector(run_id="run-1", compute_region=None)

        collector.collect()

        check_spy.assert_not_called()

    def test_collect_runs_region_check_per_bucket(self, tmp_path, mocker):
        contents = {
            "buckets": {"bucket-a": {}, "bucket-b": {}},
            "last_updated": 0.0,
        }
        mocker.patch.object(gcs_tracker, "STATS_FILE", self._write_stats_file(tmp_path, contents))

        def fake_check(bucket_name, _region, cache, _run_id):
            cache[bucket_name] = bucket_name == "bucket-a"

        mocker.patch.object(gcs_tracker, "_check_and_cache_bucket_region", side_effect=fake_check)
        collector = _make_gcs_collector(run_id="run-1", compute_region="us-east1")

        result = collector.collect()

        assert result is not None
        assert result["buckets"]["bucket-a"]["region_match"] is True
        assert result["buckets"]["bucket-b"]["region_match"] is False

    def test_collect_returns_none_on_exception(self, mocker):
        mocker.patch.object(gcs_tracker, "_read_stats_from_file", side_effect=RuntimeError("boom"))
        collector = _make_gcs_collector(run_id="run-1", compute_region=None)

        assert collector.collect() is None

    def test_region_cache_persists_across_calls(self, tmp_path, mocker):
        contents = {"buckets": {"bucket-a": {}}, "last_updated": 0.0}
        mocker.patch.object(gcs_tracker, "STATS_FILE", self._write_stats_file(tmp_path, contents))
        seen_caches: list[dict] = []

        def fake_check(bucket_name, _region, cache, _run_id):
            seen_caches.append(cache)
            cache[bucket_name] = True

        mocker.patch.object(gcs_tracker, "_check_and_cache_bucket_region", side_effect=fake_check)
        collector = _make_gcs_collector(run_id="run-1", compute_region="us-east1")

        collector.collect()
        collector.collect()

        # Same dict instance passed across calls — closure-held cache.
        assert seen_caches[0] is seen_caches[1]


class TestCollectSemaphoreStats:
    """Tests for the semaphore stats collector."""

    def test_returns_none_when_no_files(self, mocker):
        mocker.patch.object(gcs_tracker.glob, "glob", return_value=[])

        assert _collect_semaphore_stats() is None

    def test_all_types_present(self, mocker):
        mocker.patch.object(
            gcs_tracker.glob,
            "glob",
            return_value=[
                "/dev/shm/zetta_semaphore_timing_42_read",
                "/dev/shm/zetta_semaphore_timing_42_write",
                "/dev/shm/zetta_semaphore_timing_42_cuda",
                "/dev/shm/zetta_semaphore_timing_42_cpu",
                "/dev/shm/zetta_semaphore_timing_42_tensorrt",
            ],
        )

        class FakeTracker:
            def __init__(self, name, pid):
                self.name = name
                self.pid = pid

            def get_timing_data(self):
                return (1.5, 30.0, 100, 1000.0)

        mocker.patch.object(gcs_tracker, "TimingTracker", FakeTracker)

        result = _collect_semaphore_stats()

        assert result is not None
        assert set(result.keys()) == {"read", "write", "cuda", "cpu", "tensorrt"}
        assert result["cuda"] == {
            "total_wait_time": 1.5,
            "total_lease_time": 30.0,
            "lease_count": 100,
            "start_time": 1000.0,
        }

    def test_partial_types_uninitialized(self, mocker):
        mocker.patch.object(
            gcs_tracker.glob,
            "glob",
            return_value=["/dev/shm/zetta_semaphore_timing_42_read"],
        )

        class PartialTracker:
            def __init__(self, name, pid):
                self.name = name
                self.pid = pid

            def get_timing_data(self):
                if self.name in ("read", "cuda"):
                    return (0.5, 10.0, 50, 500.0)
                raise RuntimeError(f"{self.name} not initialized")

        mocker.patch.object(gcs_tracker, "TimingTracker", PartialTracker)

        result = _collect_semaphore_stats()

        assert result is not None
        assert set(result.keys()) == {"read", "cuda"}

    def test_all_types_uninitialized_returns_none(self, mocker):
        mocker.patch.object(
            gcs_tracker.glob,
            "glob",
            return_value=["/dev/shm/zetta_semaphore_timing_42_xyz"],
        )

        class FailingTracker:
            def __init__(self, name, pid):  # pylint: disable=unused-argument
                self.name = name

            def get_timing_data(self):
                raise RuntimeError("not initialized")

        mocker.patch.object(gcs_tracker, "TimingTracker", FailingTracker)

        assert _collect_semaphore_stats() is None

    def test_pid_parsed_from_filename(self, mocker):
        mocker.patch.object(
            gcs_tracker.glob,
            "glob",
            return_value=["/dev/shm/zetta_semaphore_timing_12345_read"],
        )
        seen_pids: list[int] = []

        class PidCapturingTracker:
            def __init__(self, name, pid):
                seen_pids.append(pid)
                self.name = name

            def get_timing_data(self):
                return (0.0, 0.0, 0, 0.0)

        mocker.patch.object(gcs_tracker, "TimingTracker", PidCapturingTracker)

        _collect_semaphore_stats()

        assert seen_pids and all(pid == 12345 for pid in seen_pids)

    def test_unexpected_exception_returns_none(self, mocker):
        mocker.patch.object(gcs_tracker.glob, "glob", side_effect=OSError("boom"))

        assert _collect_semaphore_stats() is None


class TestCollectResourceStats:
    """Tests for the resource stats collector."""

    def test_returns_none_when_file_missing(self, tmp_path, mocker):
        mocker.patch.object(
            gcs_tracker, "RESOURCE_STATS_FILE", str(tmp_path / "does_not_exist.json")
        )

        assert _collect_resource_stats() is None

    def test_returns_parsed_dict_from_valid_file(self, tmp_path, mocker):
        summary = {
            "cpu": {"avg_percent": 75.0, "max_percent": 95.0},
            "memory": {"total_gib": 64.0, "avg_percent": 60.0},
        }
        path = tmp_path / "resource_stats.json"
        path.write_text(json.dumps(summary))
        mocker.patch.object(gcs_tracker, "RESOURCE_STATS_FILE", str(path))

        assert _collect_resource_stats() == summary

    def test_returns_none_on_invalid_json(self, tmp_path, mocker):
        path = tmp_path / "resource_stats.json"
        path.write_text("not json at all {")
        mocker.patch.object(gcs_tracker, "RESOURCE_STATS_FILE", str(path))

        assert _collect_resource_stats() is None
