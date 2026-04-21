"""Tests for gcs_classification: bucket extraction, verb classifier, routing,
and a pricing-table-grounded fixture that locks end-to-end behaviour.

Per-row parametrization keeps the per-case noise low and makes failures
report which input row failed.
"""

import threading

import pytest

from zetta_utils.cloud_management.resource_allocation.k8s.gcs_classification import (
    BATCH_BUCKET,
    UNCLASSIFIED_BUCKET,
    GCSStats,
    classify_gcs_request,
    extract_bucket_from_api_path,
    route_request,
)

# ----------------------------------------------------------------------------
# extract_bucket_from_api_path
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path,expected",
    [
        # JSON API (object + bucket-level + upload/download/resumable prefixes).
        ("/storage/v1/b/my-bucket/o/object", "my-bucket"),
        ("/storage/v1/b/my-bucket?fields=name", "my-bucket"),
        ("/upload/storage/v1/b/my-bucket/o?uploadType=media", "my-bucket"),
        ("/download/storage/v1/b/my-bucket/o/x", "my-bucket"),
        ("/resumable/storage/v1/b/my-bucket/o/x", "my-bucket"),
        # XML API (`/{bucket}[/{object}]`) — what cloudvolume / tensorstore use.
        ("/my-bucket/path/to/object", "my-bucket"),
        ("/my-bucket/object", "my-bucket"),
        ("/my-bucket?prefix=foo", "my-bucket"),
        # Categories handled separately by the caller.
        ("/batch/storage/v1", None),
        ("/_ah/health", None),
    ],
)
def test_extract_bucket(path, expected):
    assert extract_bucket_from_api_path(path) == expected


# ----------------------------------------------------------------------------
# classify_gcs_request — JSON API
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,path,query,op_class,operation",
    [
        # Class A — writes / mutations
        ("DELETE", "/storage/v1/b/bucket/o/object", "", "A", "delete"),
        ("PATCH", "/storage/v1/b/bucket/o/object", "", "A", "patch"),
        ("POST", "/upload/storage/v1/b/bucket/o", "uploadType=media", "A", "insert"),
        ("POST", "/storage/v1/b/bucket/o", "", "A", "insert"),
        ("POST", "/storage/v1/b/bucket/o/dest/compose", "", "A", "compose"),
        (
            "POST",
            "/storage/v1/b/src/o/x/copyTo/b/dest/o/y",
            "",
            "A",
            "copy",
        ),
        (
            "POST",
            "/storage/v1/b/src/o/x/rewriteTo/b/dest/o/y",
            "",
            "A",
            "rewrite",
        ),
        ("PUT", "/upload/storage/v1/b/bucket/o/object", "uploadType=resumable", "A", "insert"),
        ("PUT", "/storage/v1/b/bucket/o/object", "", "A", "update"),
        # Class A — listings
        ("GET", "/storage/v1/b/bucket/o", "prefix=folder/", "A", "list_objects"),
        ("GET", "/storage/v1/b/bucket/o", "delimiter=/", "A", "list_objects"),
        ("GET", "/storage/v1/b/bucket/o", "maxResults=100", "A", "list_objects"),
        ("GET", "/storage/v1/b/bucket/o", "pageToken=abc", "A", "list_objects"),
        ("GET", "/storage/v1/b/bucket/o", "", "A", "list_objects"),
        ("GET", "/storage/v1/b", "", "A", "list_buckets"),
        # Class B — reads
        ("GET", "/storage/v1/b/bucket/o/path%2Fto%2Fobject", "", "B", "get"),
        ("GET", "/storage/v1/b/bucket/o/object", "alt=media", "B", "get"),
        ("HEAD", "/storage/v1/b/bucket/o/object", "", "B", "get_metadata"),
        # Unknown method
        ("OPTIONS", "/storage/v1/b/bucket/o/object", "", "B", "unknown"),
    ],
)
def test_classify_json_api(method, path, query, op_class, operation):
    assert classify_gcs_request(method, path, query) == (op_class, operation)


# ----------------------------------------------------------------------------
# classify_gcs_request — XML API
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,path,query,op_class,operation",
    [
        # Reads — must NOT be flipped to Class A by the JSON-only `/o` and `/b`
        # substring heuristics. The /output case is the regression for the
        # earlier inflation bug (`.../object`, `.../output` matched `/o`).
        ("GET", "/my-bucket/path/to/object.bin", "", "B", "get"),
        ("GET", "/my-bucket/output/file.txt", "", "B", "get"),
        ("HEAD", "/my-bucket/object", "", "B", "get_metadata"),
        # Writes
        ("PUT", "/my-bucket/path/to/object", "", "A", "insert"),
        ("DELETE", "/my-bucket/object", "", "A", "delete"),
        # Listings via XML / S3-compat query params
        ("GET", "/my-bucket/", "prefix=folder/", "A", "list_objects"),
        ("GET", "/my-bucket/", "marker=abc", "A", "list_objects"),
        ("GET", "/my-bucket/", "list-type=2", "A", "list_objects"),
        ("GET", "/my-bucket/", "max-keys=100", "A", "list_objects"),
    ],
)
def test_classify_xml_api(method, path, query, op_class, operation):
    assert classify_gcs_request(method, path, query) == (op_class, operation)


def test_classify_method_is_case_insensitive():
    assert (
        classify_gcs_request("get", "/storage/v1/b/bucket/o/object")
        == classify_gcs_request("GET", "/storage/v1/b/bucket/o/object")
        == ("B", "get")
    )


# ----------------------------------------------------------------------------
# GCSStats — record + thread safety
# ----------------------------------------------------------------------------


def test_stats_record_class_a():
    stats = GCSStats()
    stats.record("my-bucket", "A", "insert", egress_bytes=10)
    b = stats.buckets["my-bucket"]
    assert (b["class_a_count"], b["class_b_count"], b["egress_bytes"]) == (1, 0, 10)
    assert b["operations"]["insert"] == 1


def test_stats_record_class_b():
    stats = GCSStats()
    stats.record("my-bucket", "B", "get")
    b = stats.buckets["my-bucket"]
    assert (b["class_a_count"], b["class_b_count"]) == (0, 1)


def test_stats_record_uncounted():
    """op_class=None: operation+egress increment; billed totals do not."""
    stats = GCSStats()
    stats.record(UNCLASSIFIED_BUCKET, None, "unclassified", egress_bytes=42)
    b = stats.buckets[UNCLASSIFIED_BUCKET]
    assert (b["class_a_count"], b["class_b_count"], b["egress_bytes"]) == (0, 0, 42)
    assert b["operations"]["unclassified"] == 1


def test_stats_to_dict_round_trip():
    stats = GCSStats()
    stats.record("my-bucket", "A", "insert")
    stats.record("my-bucket", "B", "get")
    out = stats.to_dict()
    assert out["buckets"]["my-bucket"]["operations"] == {"insert": 1, "get": 1}
    assert "last_updated" in out


def test_stats_thread_safety():
    """Concurrent record() calls must not lose increments."""
    stats = GCSStats()
    n_threads, per_thread = 10, 100

    def worker():
        for _ in range(per_thread):
            stats.record("my-bucket", "A", "insert")
            stats.record("my-bucket", "B", "get")

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    expected = n_threads * per_thread
    assert stats.buckets["my-bucket"]["class_a_count"] == expected
    assert stats.buckets["my-bucket"]["class_b_count"] == expected


# ----------------------------------------------------------------------------
# route_request — routing decisions only (extract+classify covered above).
# ----------------------------------------------------------------------------


def test_route_real_bucket():
    stats = GCSStats()
    route_request(stats, "PUT", "/my-bucket/path/to/object")
    assert stats.buckets["my-bucket"]["class_a_count"] == 1


def test_route_query_params_split():
    """`?prefix=…` must be parsed as a list query, not a path component."""
    stats = GCSStats()
    route_request(stats, "GET", "/my-bucket/?prefix=folder")
    assert stats.buckets["my-bucket"]["operations"]["list_objects"] == 1
    assert stats.buckets["my-bucket"]["class_a_count"] == 1


def test_route_batch_path_uncounted():
    stats = GCSStats()
    route_request(stats, "POST", "/batch/storage/v1")
    b = stats.buckets[BATCH_BUCKET]
    assert b["operations"]["batch"] == 1
    assert (b["class_a_count"], b["class_b_count"]) == (0, 0)


def test_route_unrecognised_path_calls_callback():
    stats = GCSStats()
    seen: list[tuple[str, str]] = []
    route_request(stats, "GET", "/_ah/health", on_unclassified=lambda m, p: seen.append((m, p)))
    b = stats.buckets[UNCLASSIFIED_BUCKET]
    assert (b["class_a_count"], b["class_b_count"]) == (0, 0)
    assert seen == [("GET", "/_ah/health")]


# ----------------------------------------------------------------------------
# Pricing-table-grounded fixture — end-to-end through route_request.
# ----------------------------------------------------------------------------
#
# Each row is a realistic request the worker images actually send. Expected
# (bucket, class, operation) is the GCP-billed truth per
# https://cloud.google.com/storage/pricing#operations-pricing.
# Coverage assertion below: no fixture row falls through to _unclassified.
#
# Add a row whenever a new request shape is observed in production. A failure
# here = a real classifier-coverage gap, not a flaky test.

_FIXTURE = [
    # JSON API — object I/O
    ("get-json-object", "GET", "/storage/v1/b/test-bucket/o/data.bin", "test-bucket", "B", "get"),
    (
        "get-json-object-alt-media",
        "GET",
        "/storage/v1/b/test-bucket/o/data.bin?alt=media",
        "test-bucket",
        "B",
        "get",
    ),
    (
        "delete-json-object",
        "DELETE",
        "/storage/v1/b/test-bucket/o/data.bin",
        "test-bucket",
        "A",
        "delete",
    ),
    (
        "post-json-resumable-init",
        "POST",
        "/upload/storage/v1/b/test-bucket/o?uploadType=resumable",
        "test-bucket",
        "A",
        "insert",
    ),
    # JSON API — listings
    (
        "list-json-objects",
        "GET",
        "/storage/v1/b/test-bucket/o?prefix=results/",
        "test-bucket",
        "A",
        "list_objects",
    ),
    # XML API — what cloudvolume / tensorstore actually use
    (
        "get-xml-object-with-o-substring",  # regression for inflation bug
        "GET",
        "/test-bucket/output/64_64_40/0-100_0-100_0-10",
        "test-bucket",
        "B",
        "get",
    ),
    (
        "put-xml-chunk",
        "PUT",
        "/test-bucket/64_64_40/0-100_0-100_0-10",
        "test-bucket",
        "A",
        "insert",
    ),
    ("head-xml-info", "HEAD", "/test-bucket/info", "test-bucket", "B", "get_metadata"),
    (
        "list-xml-via-prefix",
        "GET",
        "/test-bucket/?prefix=64_64_40/",
        "test-bucket",
        "A",
        "list_objects",
    ),
    # Batch API — bundled sub-ops; one HTTP request, do not attribute to a class
    ("batch-storage", "POST", "/batch/storage/v1", BATCH_BUCKET, None, "batch"),
]


@pytest.mark.parametrize(
    "method,path,bucket,op_class,operation",
    [
        (method, path, bucket, op_class, operation)
        for _, method, path, bucket, op_class, operation in _FIXTURE
    ],
    ids=[label for label, *_ in _FIXTURE],
)
def test_pricing_fixture(method, path, bucket, op_class, operation):
    """End-to-end: route_request must record the fixture row under the
    expected bucket with the expected (class, operation)."""
    stats = GCSStats()
    route_request(stats, method, path)
    assert bucket in stats.buckets, f"expected bucket {bucket!r}, got {list(stats.buckets)}"
    b = stats.buckets[bucket]
    if op_class == "A":
        assert (b["class_a_count"], b["class_b_count"]) == (1, 0)
    elif op_class == "B":
        assert (b["class_a_count"], b["class_b_count"]) == (0, 1)
    else:
        # op_class=None — uncounted (e.g. _batch).
        assert (b["class_a_count"], b["class_b_count"]) == (0, 0)
    assert b["operations"][operation] == 1


def test_fixture_has_no_unclassified():
    """Coverage assertion: every fixture row must resolve cleanly. If
    `_unclassified` ever shows up, the classifier missed a real shape."""
    stats = GCSStats()
    failures: list[tuple[str, str]] = []

    def on_unclassified(method, path):
        failures.append((method, path))

    for _, method, path, *_ in _FIXTURE:
        route_request(stats, method, path, on_unclassified=on_unclassified)

    assert (
        UNCLASSIFIED_BUCKET not in stats.buckets
    ), f"classifier coverage gap — fixture rows fell through to _unclassified: {failures}"
