from datetime import datetime, timedelta, timezone

import pytest
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
