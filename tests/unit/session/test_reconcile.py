# pylint: disable=redefined-outer-name,unused-argument,import-outside-toplevel,protected-access
from datetime import datetime, timedelta, timezone

import pytest
from google.cloud.firestore_v1.base_query import FieldFilter
from kubernetes.client.exceptions import ApiException


@pytest.fixture
def mock_apis(mocker):
    batch = mocker.patch("zetta_utils.session.reconcile.k8s_client.BatchV1Api").return_value
    pod_mod = mocker.patch("zetta_utils.session.reconcile.pod", autospec=True)
    svc_mod = mocker.patch("zetta_utils.session.reconcile.service", autospec=True)
    return batch, pod_mod, svc_mod


def test_healthy_session_untouched(mock_apis, mocker):
    batch, _, _ = mock_apis
    fresh_ts = datetime.now(timezone.utc) - timedelta(minutes=5)
    mocker.patch(
        "zetta_utils.session.reconcile._query_non_down_sessions",
        return_value=[
            {
                "sessionId": "fresh-1",
                "state": "ready",
                "lastDispatchAt": fresh_ts,
                "createdAt": fresh_ts,
            }
        ],
    )
    batch.read_namespaced_job.return_value = mocker.Mock()
    write_mock = mocker.patch("zetta_utils.session.reconcile._write_session_state")

    from zetta_utils.session import reconcile

    summary = reconcile.run_reconcile()
    assert summary["reconciledCount"] == 0
    write_mock.assert_not_called()


def test_stale_but_alive_deletes_master_job(mock_apis, mocker):
    """Stale-by-time with the Job still present -> delete the master Job
    (cascade), NOT the worker resources directly."""
    batch, pod_mod, svc_mod = mock_apis
    old_ts = datetime.now(timezone.utc) - timedelta(hours=30)
    mocker.patch(
        "zetta_utils.session.reconcile._query_non_down_sessions",
        return_value=[
            {"sessionId": "old-1", "state": "ready", "lastDispatchAt": old_ts, "createdAt": old_ts}
        ],
    )
    batch.read_namespaced_job.return_value = mocker.Mock()  # Job exists
    write_mock = mocker.patch("zetta_utils.session.reconcile._write_session_state")

    from zetta_utils.session import reconcile

    summary = reconcile.run_reconcile()
    assert summary["reconciledCount"] == 1
    assert summary["staleByTime"] == 1
    assert write_mock.call_args.args[0] == "old-1"
    assert write_mock.call_args.kwargs["reason"] == "reconcile_stale_24h"
    batch.delete_namespaced_job.assert_called_once()
    assert batch.delete_namespaced_job.call_args.kwargs["name"] == "session-master-old-1"
    assert batch.delete_namespaced_job.call_args.kwargs["propagation_policy"] == "Background"
    # Cascade-GC handles the rest; reconcile does not touch worker resources.
    pod_mod.delete_namespaced_pod.assert_not_called()
    svc_mod.delete_namespaced_service.assert_not_called()


def test_master_missing_deletes_orphans(mock_apis, mocker):
    """Master Job 404 -> manually delete the orphan worker Pod, worker Service,
    and master Service; do not attempt a Job delete."""
    batch, pod_mod, svc_mod = mock_apis
    fresh_ts = datetime.now(timezone.utc) - timedelta(minutes=5)
    mocker.patch(
        "zetta_utils.session.reconcile._query_non_down_sessions",
        return_value=[
            {
                "sessionId": "orphan-1",
                "state": "ready",
                "lastDispatchAt": fresh_ts,
                "createdAt": fresh_ts,
            }
        ],
    )
    batch.read_namespaced_job.side_effect = ApiException(status=404)
    write_mock = mocker.patch("zetta_utils.session.reconcile._write_session_state")

    from zetta_utils.session import reconcile

    summary = reconcile.run_reconcile()
    assert summary["reconciledCount"] == 1
    assert summary["staleByMissingMaster"] == 1
    write_mock.assert_called_once_with("orphan-1", "down", reason="reconcile_master_missing")
    pod_mod.delete_namespaced_pod.assert_called_once()
    assert svc_mod.delete_namespaced_service.call_count == 2  # worker + master
    batch.delete_namespaced_job.assert_not_called()


def test_cleanup_failure_counts_but_does_not_crash(mock_apis, mocker):
    """A non-404 error during cleanup increments cleanupErrors, never crashes."""
    batch, _, _ = mock_apis
    old_ts = datetime.now(timezone.utc) - timedelta(hours=30)
    mocker.patch(
        "zetta_utils.session.reconcile._query_non_down_sessions",
        return_value=[
            {"sessionId": "err-1", "state": "ready", "lastDispatchAt": old_ts, "createdAt": old_ts}
        ],
    )
    batch.read_namespaced_job.return_value = mocker.Mock()  # stale-but-alive
    mocker.patch("zetta_utils.session.reconcile._write_session_state")
    batch.delete_namespaced_job.side_effect = ApiException(status=500)

    from zetta_utils.session import reconcile

    summary = reconcile.run_reconcile()
    assert summary["reconciledCount"] == 1
    assert summary["cleanupErrors"] == 1


def test_no_last_dispatch_falls_back_to_created_at(mock_apis, mocker):
    batch, _, _ = mock_apis
    old_ts = datetime.now(timezone.utc) - timedelta(hours=30)
    mocker.patch(
        "zetta_utils.session.reconcile._query_non_down_sessions",
        return_value=[
            {
                "sessionId": "never-dispatched",
                "state": "preparing",
                "lastDispatchAt": None,
                "createdAt": old_ts,
            }
        ],
    )
    batch.read_namespaced_job.return_value = mocker.Mock()
    write_mock = mocker.patch("zetta_utils.session.reconcile._write_session_state")

    from zetta_utils.session import reconcile

    summary = reconcile.run_reconcile()
    assert summary["staleByTime"] == 1
    write_mock.assert_called_once()


def test_loki_line_emitted(mock_apis, mocker, caplog):
    mocker.patch("zetta_utils.session.reconcile._query_non_down_sessions", return_value=[])
    from zetta_utils.session import reconcile

    with caplog.at_level("INFO", logger="zetta_utils.session.reconcile"):
        reconcile.run_reconcile()
    assert any("sessions.reconcile.scan_complete" in r.message for r in caplog.records)


def test_is_stale_by_time_both_none_returns_false(mocker):
    """When both lastDispatchAt and createdAt are None, _is_stale_by_time -> False."""
    from zetta_utils.session import reconcile

    result = reconcile._is_stale_by_time(
        {"lastDispatchAt": None, "createdAt": None}, datetime.now(timezone.utc)
    )
    assert result is False


def test_is_stale_by_time_old_created_at_returns_true(mocker):
    """When lastDispatchAt is None but createdAt is old, _is_stale_by_time -> True."""
    from zetta_utils.session import reconcile

    old_ts = datetime.now(timezone.utc) - timedelta(hours=30)
    result = reconcile._is_stale_by_time(
        {"lastDispatchAt": None, "createdAt": old_ts}, datetime.now(timezone.utc)
    )
    assert result is True


def test_is_master_missing_410_returns_true(mock_apis, mocker):
    """ApiException(status=410) from read_namespaced_job -> _is_master_missing True."""
    batch, _, _ = mock_apis
    batch.read_namespaced_job.side_effect = ApiException(status=410)

    from zetta_utils.session import reconcile

    assert reconcile._is_master_missing(batch, "sid") is True


def test_is_master_missing_500_reraises(mock_apis, mocker):
    """ApiException(status=500) from read_namespaced_job is re-raised."""
    batch, _, _ = mock_apis
    batch.read_namespaced_job.side_effect = ApiException(status=500)

    from zetta_utils.session import reconcile

    with pytest.raises(ApiException):
        reconcile._is_master_missing(batch, "sid")


def test_query_non_down_sessions_yields_rows(mocker):
    """_query_non_down_sessions yields rows with sessionId from each snapshot."""
    from zetta_utils.session import reconcile

    snap_a = mocker.MagicMock()
    snap_a.id = "sess-aaa"
    snap_a.to_dict.return_value = {"state": "ready"}

    snap_b = mocker.MagicMock()
    snap_b.id = "sess-bbb"
    snap_b.to_dict.return_value = None

    mock_query = mocker.MagicMock()
    mock_query.stream.return_value = [snap_a, snap_b]

    mock_collection = mocker.MagicMock()
    mock_where = mocker.MagicMock()
    mock_where.return_value = mock_query
    mock_collection.where = mock_where

    mock_db = mocker.MagicMock()
    mock_db.collection.return_value = mock_collection

    mocker.patch("zetta_utils.session.reconcile._get_sessions_db", return_value=mock_db)

    rows = list(reconcile._query_non_down_sessions())

    assert len(rows) == 2
    assert rows[0]["sessionId"] == "sess-aaa"
    assert rows[0]["state"] == "ready"
    assert rows[1]["sessionId"] == "sess-bbb"

    where_call = mock_collection.where.call_args
    passed_filter = where_call.kwargs.get("filter") or where_call.args[0]
    assert isinstance(passed_filter, FieldFilter)
    assert passed_filter.field_path == "state"
    assert passed_filter.value == "down"


def test_write_session_state_down_includes_timestamps(mocker):
    """state='down' payload includes terminatedAt and terminationReason."""
    from google.cloud import firestore

    from zetta_utils.session import reconcile

    mock_doc = mocker.MagicMock()
    mock_collection = mocker.MagicMock()
    mock_collection.document.return_value = mock_doc
    mock_db = mocker.MagicMock()
    mock_db.collection.return_value = mock_collection
    mocker.patch("zetta_utils.session.reconcile._get_sessions_db", return_value=mock_db)

    reconcile._write_session_state("sess-123", "down", reason="reconcile_stale_24h")

    mock_collection.document.assert_called_once_with("sess-123")
    set_call = mock_doc.set.call_args
    payload = set_call.args[0]
    assert payload["state"] == "down"
    assert payload["terminationReason"] == "reconcile_stale_24h"
    assert payload["terminatedAt"] is firestore.SERVER_TIMESTAMP
    assert set_call.kwargs.get("merge") is True


def test_write_session_state_non_down_minimal_payload(mocker):
    """state != 'down' payload contains only the state key."""
    from zetta_utils.session import reconcile

    mock_doc = mocker.MagicMock()
    mock_collection = mocker.MagicMock()
    mock_collection.document.return_value = mock_doc
    mock_db = mocker.MagicMock()
    mock_db.collection.return_value = mock_collection
    mocker.patch("zetta_utils.session.reconcile._get_sessions_db", return_value=mock_db)

    reconcile._write_session_state("sess-456", "ready", reason="unused")

    set_call = mock_doc.set.call_args
    payload = set_call.args[0]
    assert payload == {"state": "ready"}
    assert set_call.kwargs.get("merge") is True
