import asyncio

import aiohttp
import pytest
from fastapi import HTTPException
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
    mocker.patch("zetta_utils.session.master._check_zetta_ai_token")
    master._worker_endpoint = "http://session-worker-test/"

    result = await master.status(authorization="Bearer fake@zetta.ai")
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


async def test_h_auth_middleware_rejects_non_zetta_token(master_env, mocker):
    """/dispatch rejects tokens not @zetta.ai."""
    from zetta_utils.session import master

    mocker.patch(
        "zetta_utils.session.master._check_zetta_ai_token",
        side_effect=PermissionError("not @zetta.ai"),
    )
    with pytest.raises(PermissionError):
        await master.dispatch(
            body=master.DispatchBody(
                specUrl="gs://x", runId="r", jobType="j", requiredPreload="try"
            ),
            authorization="Bearer external@gmail.com",
        )


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

    mocker.patch("zetta_utils.session.master._check_zetta_ai_token")
    mocker.patch("zetta_utils.session.master._update_last_dispatch_at")
    aiohttp_mock_session.set_post_response(
        status=200,
        json_payload={"result": "ok", "dispatchCount": 1, "nearRecycle": False},
    )

    body = master.DispatchBody(specUrl="gs://x", runId="r1", jobType="j", requiredPreload="try")
    results = await asyncio.gather(
        master.dispatch(body=body, authorization="Bearer fake@zetta.ai"),
        master.dispatch(body=body, authorization="Bearer fake@zetta.ai"),
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

    with pytest.raises(HTTPException) as exc_info:
        await master._forward_dispatch_to_worker(
            {
                "specUrl": "gs://x",
                "runId": "r1",
                "jobType": "j",
                "requiredPreload": "try",
            }
        )
    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "worker_run_spec_error"
    recreate_spy.assert_not_called()
    assert aiohttp_mock_session.post.call_count == 2
