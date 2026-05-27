# pylint: disable=protected-access,unused-argument,import-outside-toplevel
import asyncio
import json

import aiohttp
import pytest
from aiohttp import web
from kubernetes.client.exceptions import ApiException


async def test_a_boot_creates_pod_and_service_with_owner_refs(
    master_env, mock_k8s_apis, mocker, aiohttp_mock_session
):
    """Boot creates Pod + Service with correct ownerReferences."""
    core_mock = mock_k8s_apis
    mocker.patch(
        "zetta_utils.session.master._read_session_row",
        return_value={
            "state": "preparing",
            "initialPreload": "try",
            "config": {"idleTtlSec": 60},
        },
    )
    mocker.patch("zetta_utils.session.master._read_queue_docs", return_value=[])
    mocker.patch("zetta_utils.session.master._write_session_state")
    aiohttp_mock_session.set_get_response(status=200)

    from zetta_utils.session import master

    await master._boot()

    pod_call = core_mock.create_namespaced_pod.call_args
    pod_body = pod_call.kwargs["body"]
    assert pod_body["spec"]["automountServiceAccountToken"] is False
    assert pod_body["metadata"]["ownerReferences"][0]["uid"] == "pod-uid-xyz"

    svc_call = core_mock.create_namespaced_service.call_args
    svc_body = svc_call.kwargs["body"]
    assert svc_body["metadata"]["name"] == "session-worker-test-uuid-001"
    assert svc_body["metadata"]["namespace"] == "sessions"
    assert svc_body["metadata"]["ownerReferences"][0]["uid"] == "pod-uid-xyz"
    assert svc_body["spec"]["selector"]["sessionId"] == "test-uuid-001"
    assert svc_body["spec"]["ports"][0]["port"] == 80
    assert svc_body["spec"]["ports"][0]["targetPort"] == 80


async def test_b_idle_timer_fires_after_ttl(master_env, mock_k8s_apis, mocker):
    """Idle timer fires after idleTtlSec."""
    fired = asyncio.Event()

    async def _on_shutdown_capture(*, reason: str) -> None:
        fired.set()

    mocker.patch(
        "zetta_utils.session.master._on_shutdown",
        side_effect=_on_shutdown_capture,
    )
    from zetta_utils.session import master

    master._idle_ttl_sec = 0.01
    master._start_idle_timer()
    await asyncio.wait_for(fired.wait(), timeout=2.0)
    assert fired.is_set()


async def test_c_idle_timer_cancels_on_new_dispatch(master_env, mock_k8s_apis, mocker):
    """Idle timer cancels on dispatch arrival."""
    from zetta_utils.session import master

    master._idle_ttl_sec = 0.1
    master._start_idle_timer()
    master._cancel_idle_timer()
    await asyncio.sleep(0.15)
    assert master._idle_timer_task is None


async def test_d_recycle_on_near_recycle_then_refused(
    master_env, mock_k8s_apis, mocker, aiohttp_mock_session
):
    """On connection-refused after nearRecycle=true, worker is recreated."""
    from zetta_utils.session import master

    master._last_response_near_recycle = True
    master._worker_endpoint = "http://session-worker-test/"

    mocker.patch("zetta_utils.session.master._update_last_dispatch_at")
    success_cm, _ = aiohttp_mock_session._make_response(
        status=200,
        json_payload={"result": 42, "dispatchCount": 1, "nearRecycle": False},
    )
    aiohttp_mock_session.post.side_effect = [
        aiohttp.ClientConnectionError("dead"),
        success_cm,
    ]
    recreate_mock = mocker.patch(
        "zetta_utils.session.master._recreate_worker_pod",
        new_callable=mocker.AsyncMock,
    )

    body = await master._forward_dispatch_to_worker(
        {"specUrl": "gs://x", "runId": "r1", "jobType": "j", "requiredPreload": "try"}
    )
    recreate_mock.assert_called_once()
    assert body["dispatchCount"] == 1


async def test_e_terminate_cleans_up(master_env, mock_k8s_apis, mocker):
    """Terminate path deletes Pod + Service and writes state=down."""
    core_mock = mock_k8s_apis
    write_mock = mocker.patch("zetta_utils.session.master._write_session_state")

    from zetta_utils.session import master

    await master._on_shutdown(reason="explicit_terminate")

    core_mock.delete_namespaced_pod.assert_called()
    core_mock.delete_namespaced_service.assert_called()
    write_mock.assert_called_once_with("down", reason="explicit_terminate")


async def test_f_worker_404_writes_down_gracefully(
    master_env, mock_k8s_apis, mocker, aiohttp_mock_session
):
    """Worker Pod 404 on status poll -> state=down, no crash."""
    core_mock = mock_k8s_apis
    core_mock.read_namespaced_pod_status.side_effect = ApiException(status=404)
    aiohttp_mock_session.get.side_effect = aiohttp.ClientConnectionError("gone")

    from zetta_utils.session import master

    write_mock = mocker.patch("zetta_utils.session.master._write_session_state")
    master._worker_endpoint = "http://session-worker-test/"

    result = await master._status_logic()
    assert result["state"] == "down"
    write_mock.assert_called_with("down", reason="proxy_unreachable")


async def test_g_queue_drain_polls_until_empty(
    master_env, mock_k8s_apis, mocker, aiohttp_mock_session
):
    """Drain polls until empty (covers the write-read race)."""
    read_mock = mocker.patch(
        "zetta_utils.session.master._read_queue_docs",
        side_effect=[
            [
                {
                    "dispatchId": "d1",
                    "specUrl": "gs://1",
                    "runId": "r1",
                    "jobType": "j",
                    "requiredPreload": "try",
                }
            ],
            [
                {
                    "dispatchId": "d2",
                    "specUrl": "gs://2",
                    "runId": "r2",
                    "jobType": "j",
                    "requiredPreload": "try",
                }
            ],
            [],
            [],
        ],
    )
    mocker.patch("zetta_utils.session.master._delete_queue_doc")
    mocker.patch("zetta_utils.session.master._update_last_dispatch_at")
    aiohttp_mock_session.set_post_response(
        status=200,
        json_payload={"result": None, "dispatchCount": 1, "nearRecycle": False},
    )

    from zetta_utils.session import master

    master._worker_endpoint = "http://session-worker-test/"
    await master._drain_pre_ready_queue()

    assert read_mock.call_count >= 3


async def test_i_phase_failed_with_nonzero_exit_is_permanent(master_env, mock_k8s_apis, mocker):
    """phase=Failed + exitCode=1 -> permanent self-check failure."""
    core_mock = mock_k8s_apis
    from kubernetes.client import (
        V1ContainerState,
        V1ContainerStateTerminated,
        V1ContainerStatus,
        V1Pod,
        V1PodStatus,
    )

    core_mock.read_namespaced_pod_status.return_value = V1Pod(
        status=V1PodStatus(
            phase="Failed",
            container_statuses=[
                V1ContainerStatus(
                    name="session-worker",
                    image="x",
                    image_id="x",
                    ready=False,
                    restart_count=0,
                    state=V1ContainerState(terminated=V1ContainerStateTerminated(exit_code=1)),
                )
            ],
        )
    )
    mocker.patch("zetta_utils.session.master._terminate_session", new_callable=mocker.AsyncMock)
    from zetta_utils.session import master

    verdict = master._classify_worker_failure()
    assert verdict == "permanent"


async def test_j_phase_pending_is_transient_under_cap(master_env, mock_k8s_apis, mocker):
    """phase=Pending -> transient; keep polling within the 60s budget."""
    core_mock = mock_k8s_apis
    from kubernetes.client import V1Pod, V1PodStatus

    core_mock.read_namespaced_pod_status.return_value = V1Pod(status=V1PodStatus(phase="Pending"))
    from zetta_utils.session import master

    verdict = master._classify_worker_failure()
    assert verdict == "transient"


async def test_k_sigterm_during_boot_is_safe(
    master_env, mock_k8s_apis, mocker, aiohttp_mock_session
):
    """SIGTERM during _wait_for_worker_healthz must run _on_shutdown cleanly."""
    core_mock = mock_k8s_apis
    mocker.patch(
        "zetta_utils.session.master._read_session_row",
        return_value={
            "state": "preparing",
            "initialPreload": "try",
            "config": {"idleTtlSec": 60},
        },
    )
    mocker.patch("zetta_utils.session.master._read_queue_docs", return_value=[])
    mocker.patch("zetta_utils.session.master._write_session_state")

    aiohttp_mock_session.get.side_effect = aiohttp.ClientConnectionError("never ready")

    from zetta_utils.session import master

    boot_task = asyncio.create_task(master._boot())
    await asyncio.sleep(0.05)

    await master._on_shutdown(reason="explicit_terminate")

    assert core_mock.delete_namespaced_pod.call_count == 1
    assert core_mock.delete_namespaced_service.call_count == 1

    boot_task.cancel()
    with pytest.raises((asyncio.CancelledError, aiohttp.ClientConnectionError, SystemExit)):
        await boot_task


async def test_l_concurrent_dispatches_idle_timer_safe(
    master_env, mock_k8s_apis, mocker, aiohttp_mock_session
):
    """Two concurrent dispatches must not leave an orphan idle-timer task."""
    from zetta_utils.session import master

    master._idle_ttl_sec = 60
    master._worker_endpoint = "http://session-worker-test/"

    mocker.patch("zetta_utils.session.master._update_last_dispatch_at")
    aiohttp_mock_session.set_post_response(
        status=200,
        json_payload={"result": "ok", "dispatchCount": 1, "nearRecycle": False},
    )

    body = {"specUrl": "gs://x", "runId": "r1", "jobType": "j", "requiredPreload": "try"}
    results = await asyncio.gather(
        master._dispatch_logic(body, authorization="Bearer fake@zetta.ai"),
        master._dispatch_logic(body, authorization="Bearer fake@zetta.ai"),
    )
    assert all(r["dispatchCount"] == 1 for r in results)

    assert master._idle_timer_task is not None
    assert not master._idle_timer_task.done()


async def test_m_worker_500_surfaces_as_502_no_recycle(
    master_env, mock_k8s_apis, mocker, aiohttp_mock_session
):
    """Worker /run_spec/ HTTP 500 -> master returns 502 after one bounded retry."""
    from zetta_utils.session import master

    master._last_response_near_recycle = False
    master._worker_endpoint = "http://session-worker-test/"

    mocker.patch("asyncio.sleep", new_callable=mocker.AsyncMock)
    recreate_spy = mocker.patch(
        "zetta_utils.session.master._recreate_worker_pod",
        new_callable=mocker.AsyncMock,
    )

    def _raise_500(*_, **__):
        request_info = mocker.MagicMock()
        history: tuple = ()
        raise aiohttp.ClientResponseError(
            request_info=request_info,
            history=history,
            status=500,
            message="worker_500",
            headers=None,
        )

    aiohttp_mock_session.set_post_response(status=500)
    aiohttp_mock_session.post.return_value.__aenter__.return_value.raise_for_status = (
        mocker.MagicMock(side_effect=_raise_500)
    )

    with pytest.raises(web.HTTPBadGateway) as exc_info:
        await master._forward_dispatch_to_worker(
            {
                "specUrl": "gs://x",
                "runId": "r1",
                "jobType": "j",
                "requiredPreload": "try",
            }
        )
    assert exc_info.value.status == 502
    assert exc_info.value.reason == "worker_run_spec_error"
    recreate_spy.assert_not_called()
    assert aiohttp_mock_session.post.call_count == 2


# ---- Firestore helpers --------------------------------------------------


async def test_read_session_row_returns_dict(mocker):
    """_read_session_row returns the snapshot dict via the sessions chain."""
    from zetta_utils.session import master

    db = mocker.MagicMock()
    snapshot = mocker.MagicMock()
    snapshot.to_dict.return_value = {"state": "ready"}
    db.collection.return_value.document.return_value.get.return_value = snapshot
    mocker.patch("zetta_utils.session.master._get_sessions_db", return_value=db)

    result = master._read_session_row("sid-1")

    assert result == {"state": "ready"}
    db.collection.assert_called_once_with("sessions")
    db.collection.return_value.document.assert_called_once_with("sid-1")
    db.collection.return_value.document.return_value.get.assert_called_once_with()


async def test_read_session_row_none_to_dict_returns_empty(mocker):
    """_read_session_row returns {} when to_dict() is None."""
    from zetta_utils.session import master

    db = mocker.MagicMock()
    snapshot = mocker.MagicMock()
    snapshot.to_dict.return_value = None
    db.collection.return_value.document.return_value.get.return_value = snapshot
    mocker.patch("zetta_utils.session.master._get_sessions_db", return_value=db)

    assert master._read_session_row("sid-1") == {}


async def test_write_session_state_plain(master_env, mocker):
    """_write_session_state merges {'state': s} for a non-down state."""
    from zetta_utils.session import master

    db = mocker.MagicMock()
    mocker.patch("zetta_utils.session.master._get_sessions_db", return_value=db)

    master._write_session_state("ready")

    doc = db.collection.return_value.document.return_value
    doc.set.assert_called_once_with({"state": "ready"}, merge=True)


async def test_write_session_state_down_with_reason(master_env, mocker):
    """state=down with a reason stamps terminatedAt and terminationReason."""
    from datetime import datetime

    from zetta_utils.session import master

    db = mocker.MagicMock()
    mocker.patch("zetta_utils.session.master._get_sessions_db", return_value=db)

    master._write_session_state("down", reason="boom")

    doc = db.collection.return_value.document.return_value
    payload = doc.set.call_args.args[0]
    assert doc.set.call_args.kwargs == {"merge": True}
    assert payload["state"] == "down"
    assert isinstance(payload["terminatedAt"], datetime)
    assert payload["terminationReason"] == "boom"


async def test_write_session_state_down_no_reason(master_env, mocker):
    """state=down with reason=None omits the terminationReason key."""
    from zetta_utils.session import master

    db = mocker.MagicMock()
    mocker.patch("zetta_utils.session.master._get_sessions_db", return_value=db)

    master._write_session_state("down")

    doc = db.collection.return_value.document.return_value
    payload = doc.set.call_args.args[0]
    assert payload["state"] == "down"
    assert "terminatedAt" in payload
    assert "terminationReason" not in payload


async def test_read_queue_docs_orders_and_stamps_dispatch_id(mocker):
    """_read_queue_docs orders by enqueuedAt asc and stamps dispatchId."""
    from google.cloud import firestore

    from zetta_utils.session import master

    db = mocker.MagicMock()
    snap_a = mocker.MagicMock()
    snap_a.id = "d-a"
    snap_a.to_dict.return_value = {"specUrl": "gs://a"}
    snap_b = mocker.MagicMock()
    snap_b.id = "d-b"
    snap_b.to_dict.return_value = None
    query = db.collection.return_value.document.return_value.collection.return_value.order_by
    query.return_value.stream.return_value = [snap_a, snap_b]
    mocker.patch("zetta_utils.session.master._get_sessions_db", return_value=db)

    docs = master._read_queue_docs("sid-1")

    query.assert_called_once_with("enqueuedAt", direction=firestore.Query.ASCENDING)
    assert docs[0]["dispatchId"] == "d-a"
    assert docs[0]["specUrl"] == "gs://a"
    assert docs[1]["dispatchId"] == "d-b"


async def test_delete_queue_doc_chain(mocker):
    """_delete_queue_doc deletes the queue document by dispatch id."""
    from zetta_utils.session import master

    db = mocker.MagicMock()
    mocker.patch("zetta_utils.session.master._get_sessions_db", return_value=db)

    master._delete_queue_doc("sid-1", "d-1")

    sessions = db.collection.return_value
    sessions.document.assert_called_once_with("sid-1")
    queue = sessions.document.return_value.collection.return_value
    queue.document.assert_called_once_with("d-1")
    queue.document.return_value.delete.assert_called_once_with()


async def test_update_last_dispatch_at(master_env, mocker):
    """_update_last_dispatch_at merges a server-timestamp lastDispatchAt."""
    from google.cloud import firestore

    from zetta_utils.session import master

    db = mocker.MagicMock()
    mocker.patch("zetta_utils.session.master._get_sessions_db", return_value=db)

    master._update_last_dispatch_at()

    doc = db.collection.return_value.document.return_value
    doc.set.assert_called_once_with({"lastDispatchAt": firestore.SERVER_TIMESTAMP}, merge=True)


# ---- main() -------------------------------------------------------------


async def test_main_runs_full_lifecycle(master_env, mocker):
    """main() installs the handler, boots, serves, then shuts down once."""
    from zetta_utils.session import master

    sigterm = mocker.patch("zetta_utils.session.master._install_sigterm_handler")
    boot = mocker.patch("zetta_utils.session.master._boot", new_callable=mocker.AsyncMock)
    serve = mocker.patch(
        "zetta_utils.session.master._serve_forever", new_callable=mocker.AsyncMock
    )
    on_shutdown = mocker.patch(
        "zetta_utils.session.master._on_shutdown", new_callable=mocker.AsyncMock
    )

    await master.main()

    sigterm.assert_called_once_with()
    boot.assert_awaited_once_with()
    serve.assert_awaited_once_with()
    on_shutdown.assert_awaited_once_with(reason="explicit_terminate")


async def test_main_shuts_down_when_boot_raises(master_env, mocker):
    """main() runs _on_shutdown in finally even if _boot raises."""
    from zetta_utils.session import master

    mocker.patch("zetta_utils.session.master._install_sigterm_handler")
    mocker.patch(
        "zetta_utils.session.master._boot",
        new_callable=mocker.AsyncMock,
        side_effect=RuntimeError("boom"),
    )
    serve = mocker.patch(
        "zetta_utils.session.master._serve_forever", new_callable=mocker.AsyncMock
    )
    on_shutdown = mocker.patch(
        "zetta_utils.session.master._on_shutdown", new_callable=mocker.AsyncMock
    )

    with pytest.raises(RuntimeError):
        await master.main()

    serve.assert_not_called()
    on_shutdown.assert_awaited_once_with(reason="explicit_terminate")


# ---- Boot / classify ----------------------------------------------------


async def test_boot_unexpected_state_exits(master_env, mocker):
    """_boot exits with code 2 when the session is not 'preparing'."""
    from zetta_utils.session import master

    mocker.patch(
        "zetta_utils.session.master._read_session_row",
        return_value={"state": "ready"},
    )

    with pytest.raises(SystemExit) as exc_info:
        await master._boot()
    assert exc_info.value.code == 2


async def test_classify_worker_failure_404_permanent(master_env, mock_k8s_apis):
    """A 404 on read_namespaced_pod_status classifies as permanent."""
    mock_k8s_apis.read_namespaced_pod_status.side_effect = ApiException(status=404)

    from zetta_utils.session import master

    assert master._classify_worker_failure() == "permanent"


async def test_classify_worker_failure_500_reraises(master_env, mock_k8s_apis):
    """A non-404 ApiException re-raises out of _classify_worker_failure."""
    mock_k8s_apis.read_namespaced_pod_status.side_effect = ApiException(status=500)

    from zetta_utils.session import master

    with pytest.raises(ApiException):
        master._classify_worker_failure()


async def test_classify_worker_failure_succeeded_permanent(master_env, mock_k8s_apis):
    """phase=Succeeded classifies as permanent (worker is gone)."""
    from kubernetes.client import V1Pod, V1PodStatus

    mock_k8s_apis.read_namespaced_pod_status.return_value = V1Pod(
        status=V1PodStatus(phase="Succeeded")
    )

    from zetta_utils.session import master

    assert master._classify_worker_failure() == "permanent"


async def test_classify_worker_failure_unknown_permanent(master_env, mock_k8s_apis):
    """phase=None classifies as permanent."""
    from kubernetes.client import V1Pod, V1PodStatus

    mock_k8s_apis.read_namespaced_pod_status.return_value = V1Pod(status=V1PodStatus(phase=None))

    from zetta_utils.session import master

    assert master._classify_worker_failure() == "permanent"


async def test_wait_for_worker_healthz_timeout(
    master_env, mock_k8s_apis, mocker, aiohttp_mock_session
):
    """An expired deadline terminates with worker_healthz_timeout."""
    mocker.patch("zetta_utils.session.master.WORKER_HEALTHZ_TIMEOUT_S", -1)
    terminate = mocker.patch(
        "zetta_utils.session.master._terminate_session",
        new_callable=mocker.AsyncMock,
        side_effect=SystemExit,
    )

    from zetta_utils.session import master

    master._worker_endpoint = "http://session-worker-test/"

    with pytest.raises(SystemExit):
        await master._wait_for_worker_healthz()
    terminate.assert_awaited_with("worker_healthz_timeout")


async def test_wait_for_worker_healthz_permanent_failure(
    master_env, mock_k8s_apis, mocker, aiohttp_mock_session
):
    """Repeated refusals + permanent verdict terminates the boot self-check."""
    aiohttp_mock_session.get.side_effect = aiohttp.ClientConnectionError("dead")
    mocker.patch(
        "zetta_utils.session.master._classify_worker_failure",
        return_value="permanent",
    )
    mocker.patch("asyncio.sleep", new_callable=mocker.AsyncMock)
    terminate = mocker.patch(
        "zetta_utils.session.master._terminate_session",
        new_callable=mocker.AsyncMock,
        side_effect=SystemExit,
    )

    from zetta_utils.session import master

    master._worker_endpoint = "http://session-worker-test/"

    with pytest.raises(SystemExit):
        await master._wait_for_worker_healthz()
    terminate.assert_awaited_with("worker_boot_self_check_failed")


# ---- Terminate / shutdown ----------------------------------------------


async def test_terminate_session_nonidle_exit_code_1(master_env, mocker):
    """_terminate_session with a non-idle reason exits 1 and shuts down."""
    from zetta_utils.session import master

    on_shutdown = mocker.patch(
        "zetta_utils.session.master._on_shutdown", new_callable=mocker.AsyncMock
    )

    with pytest.raises(SystemExit) as exc_info:
        await master._terminate_session("boom")
    assert exc_info.value.code == 1
    on_shutdown.assert_awaited_once_with(reason="boom")


async def test_terminate_session_idle_exit_code_0(master_env, mocker):
    """_terminate_session with reason=idle_timer exits 0."""
    from zetta_utils.session import master

    mocker.patch("zetta_utils.session.master._on_shutdown", new_callable=mocker.AsyncMock)

    with pytest.raises(SystemExit) as exc_info:
        await master._terminate_session("idle_timer")
    assert exc_info.value.code == 0


async def test_on_shutdown_idempotent(master_env, mock_k8s_apis, mocker):
    """_on_shutdown returns early when shutdown already started."""
    from zetta_utils.session import master

    master._shutdown_started = True
    write_mock = mocker.patch("zetta_utils.session.master._write_session_state")

    await master._on_shutdown(reason="x")

    write_mock.assert_not_called()
    mock_k8s_apis.delete_namespaced_pod.assert_not_called()


# ---- Handlers / build_app ----------------------------------------------


async def test_dispatch_handler_assembles_payload(master_env, mocker):
    """dispatch handler forwards the assembled payload and authorization."""
    from zetta_utils.session import master

    request = mocker.MagicMock()
    request.json = mocker.AsyncMock(
        return_value={
            "specUrl": "gs://x",
            "runId": "r1",
            "jobType": "j",
            "requiredPreload": "x",
        }
    )
    request.headers = {"Authorization": "Bearer t"}
    logic = mocker.patch(
        "zetta_utils.session.master._dispatch_logic",
        new_callable=mocker.AsyncMock,
        return_value={"ok": 1},
    )

    resp = await master.dispatch(request)

    logic.assert_awaited_once_with(
        {
            "specUrl": "gs://x",
            "runId": "r1",
            "jobType": "j",
            "requiredPreload": "x",
        },
        authorization="Bearer t",
    )
    assert json.loads(resp.body) == {"ok": 1}


async def test_status_handler_returns_logic_result(master_env, mocker):
    """status handler returns the _status_logic result as JSON."""
    from zetta_utils.session import master

    mocker.patch(
        "zetta_utils.session.master._status_logic",
        new_callable=mocker.AsyncMock,
        return_value={"state": "ready"},
    )

    resp = await master.status(mocker.MagicMock())
    assert json.loads(resp.body) == {"state": "ready"}


async def test_terminate_handler_returns_logic_result(master_env, mocker):
    """terminate handler returns the _terminate_logic result as JSON."""
    from zetta_utils.session import master

    mocker.patch(
        "zetta_utils.session.master._terminate_logic",
        new_callable=mocker.AsyncMock,
        return_value={"state": "down"},
    )

    resp = await master.terminate(mocker.MagicMock())
    assert json.loads(resp.body) == {"state": "down"}


async def test_build_app_registers_routes(master_env):
    """_build_app registers POST /dispatch, GET /status, POST /terminate."""
    from zetta_utils.session import master

    app = master._build_app()
    registered = {(route.method, route.resource.canonical) for route in app.router.routes()}
    assert ("POST", "/dispatch") in registered
    assert ("GET", "/status") in registered
    assert ("POST", "/terminate") in registered


# ---- Endpoint logic -----------------------------------------------------


async def test_status_logic_healthy(master_env, aiohttp_mock_session):
    """_status_logic returns ready on a 200 healthz response."""
    aiohttp_mock_session.set_get_response(status=200)

    from zetta_utils.session import master

    master._worker_endpoint = "http://session-worker-test/"
    assert await master._status_logic() == {"state": "ready"}


async def test_status_logic_unhealthy(master_env, aiohttp_mock_session):
    """_status_logic returns down on a 503 healthz response."""
    aiohttp_mock_session.set_get_response(status=503)

    from zetta_utils.session import master

    master._worker_endpoint = "http://session-worker-test/"
    assert await master._status_logic() == {"state": "down"}


async def test_terminate_logic_shuts_down_and_stops(master_env, mocker):
    """_terminate_logic shuts down, requests serve-stop, returns down."""
    from zetta_utils.session import master

    on_shutdown = mocker.patch(
        "zetta_utils.session.master._on_shutdown", new_callable=mocker.AsyncMock
    )
    stop = mocker.patch("zetta_utils.session.master._request_serve_stop")

    result = await master._terminate_logic()

    assert result == {"state": "down"}
    on_shutdown.assert_awaited_once_with(reason="explicit_terminate")
    stop.assert_called_once_with()


# ---- Forwarding to worker ----------------------------------------------


async def test_forward_dispatch_bare_token_gets_bearer_prefix(
    master_env, mocker, aiohttp_mock_session
):
    """A bare user token is prefixed with 'Bearer ' on the worker request."""
    from zetta_utils.session import master

    master._worker_endpoint = "http://session-worker-test/"
    aiohttp_mock_session.set_post_response(
        status=200, json_payload={"dispatchCount": 1, "nearRecycle": False}
    )
    mocker.patch("zetta_utils.session.master._update_last_dispatch_at")

    await master._forward_dispatch_to_worker(
        {"specUrl": "gs://x", "runId": "r1", "jobType": "j", "requiredPreload": "try"},
        user_token="rawtoken",
    )

    headers = aiohttp_mock_session.post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer rawtoken"


async def test_forward_dispatch_4xx_no_retry(master_env, mocker, aiohttp_mock_session):
    """A worker 4xx surfaces as a 502 client-error gateway with no retry."""
    from zetta_utils.session import master

    master._worker_endpoint = "http://session-worker-test/"

    def _raise_404(*_, **__):
        raise aiohttp.ClientResponseError(
            request_info=mocker.MagicMock(),
            history=(),
            status=404,
            message="worker_404",
            headers=None,
        )

    aiohttp_mock_session.set_post_response(status=404)
    aiohttp_mock_session.post.return_value.__aenter__.return_value.raise_for_status = (
        mocker.MagicMock(side_effect=_raise_404)
    )

    with pytest.raises(web.HTTPBadGateway) as exc_info:
        await master._forward_dispatch_to_worker(
            {"specUrl": "gs://x", "runId": "r1", "jobType": "j", "requiredPreload": "try"}
        )
    assert exc_info.value.status == 502
    assert exc_info.value.reason == "worker_run_spec_client_error"
    assert aiohttp_mock_session.post.call_count == 1


async def test_forward_dispatch_conn_refused_permanent(master_env, mocker, aiohttp_mock_session):
    """Connection refused + permanent verdict terminates then raises 502."""
    from zetta_utils.session import master

    master._last_response_near_recycle = False
    master._worker_endpoint = "http://session-worker-test/"
    aiohttp_mock_session.post.side_effect = aiohttp.ClientConnectionError("dead")
    mocker.patch(
        "zetta_utils.session.master._classify_worker_failure",
        return_value="permanent",
    )
    terminate = mocker.patch(
        "zetta_utils.session.master._terminate_session",
        new_callable=mocker.AsyncMock,
    )

    with pytest.raises(web.HTTPBadGateway) as exc_info:
        await master._forward_dispatch_to_worker(
            {"specUrl": "gs://x", "runId": "r1", "jobType": "j", "requiredPreload": "try"}
        )
    assert exc_info.value.reason == "worker unreachable"
    terminate.assert_awaited_with("worker_boot_self_check_failed")


async def test_forward_dispatch_conn_refused_transient(master_env, mocker, aiohttp_mock_session):
    """Connection refused + transient verdict raises 502 without terminating."""
    from zetta_utils.session import master

    master._last_response_near_recycle = False
    master._worker_endpoint = "http://session-worker-test/"
    aiohttp_mock_session.post.side_effect = aiohttp.ClientConnectionError("dead")
    mocker.patch(
        "zetta_utils.session.master._classify_worker_failure",
        return_value="transient",
    )
    terminate = mocker.patch(
        "zetta_utils.session.master._terminate_session",
        new_callable=mocker.AsyncMock,
    )

    with pytest.raises(web.HTTPBadGateway) as exc_info:
        await master._forward_dispatch_to_worker(
            {"specUrl": "gs://x", "runId": "r1", "jobType": "j", "requiredPreload": "try"}
        )
    assert exc_info.value.reason == "worker unreachable"
    terminate.assert_not_called()
    assert aiohttp_mock_session.post.call_count == 1


# ---- Recycle / idle timer ----------------------------------------------


async def test_recreate_worker_pod_resets_state(master_env, mock_k8s_apis, mocker):
    """_recreate_worker_pod recreates the Pod and resets recycle state."""
    from zetta_utils.session import master

    master._last_response_near_recycle = True
    master._active_dispatch_count_on_worker = 5
    mocker.patch(
        "zetta_utils.session.master._read_session_row",
        return_value={"initialPreload": "try"},
    )
    wait = mocker.patch(
        "zetta_utils.session.master._wait_for_worker_healthz",
        new_callable=mocker.AsyncMock,
    )
    mocker.patch("asyncio.sleep", new_callable=mocker.AsyncMock)

    await master._recreate_worker_pod()

    mock_k8s_apis.delete_namespaced_pod.assert_called_once()
    assert (
        mock_k8s_apis.delete_namespaced_pod.call_args.kwargs["name"]
        == "session-worker-test-uuid-001"
    )
    mock_k8s_apis.create_namespaced_pod.assert_called_once()
    wait.assert_awaited_once_with()
    assert master._last_response_near_recycle is False
    assert master._active_dispatch_count_on_worker == 0


async def test_start_idle_timer_is_idempotent(master_env, mocker):
    """A second _start_idle_timer call does not replace the live task."""
    from zetta_utils.session import master

    master._idle_ttl_sec = 60
    master._start_idle_timer()
    task1 = master._idle_timer_task
    master._start_idle_timer()
    try:
        assert master._idle_timer_task is task1
    finally:
        master._cancel_idle_timer()
