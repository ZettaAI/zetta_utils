import asyncio

import pytest

pytest.importorskip("fastapi")

import aiohttp
from fastapi import HTTPException
from kubernetes.client.exceptions import ApiException


async def test_create_session_reserves_and_creates_job_and_service(
    manager_env, mock_batch_v1, mocker
):
    # Reservation succeeds (no concurrent/pre-existing session) -> returns None.
    reserve = mocker.patch("web_api.app.session._reserve_or_get_existing", return_value=None)
    svc_create = mocker.patch("web_api.app.session.service.create_namespaced_service")
    mock_batch_v1.create_namespaced_job.return_value = mocker.Mock(
        metadata=mocker.Mock(uid="job-uid-1")
    )

    from web_api.app import session

    resp = await session.create_session(
        body=session.CreateSessionBody(ownerType="t", ownerId="o", initialPreload="try"),
    )
    assert resp.state == "preparing"
    assert resp.controlEndpoint.startswith("http://session-master-")
    assert resp.controlEndpoint.endswith(".sessions.svc.cluster.local/")

    # The reserved row carried into the transaction is well-formed.
    reserved_row = reserve.call_args.args[3]
    assert reserved_row["state"] == "preparing"
    assert reserved_row["ownerType"] == "t"

    job_body = mock_batch_v1.create_namespaced_job.call_args.kwargs["body"]
    assert job_body["metadata"]["namespace"] == "sessions"
    assert resp.sessionId in job_body["metadata"]["name"]

    svc_body = svc_create.call_args.kwargs["body"]
    assert svc_body["metadata"]["ownerReferences"][0]["uid"] == "job-uid-1"
    assert svc_body["spec"]["selector"]["sessionId"] == resp.sessionId


async def test_create_session_reuses_active(manager_env, mock_batch_v1, mocker):
    # A concurrent/pre-existing active session is returned by the transaction.
    mocker.patch(
        "web_api.app.session._reserve_or_get_existing",
        return_value={"sessionId": "existing-uuid", "state": "ready"},
    )
    from web_api.app import session

    resp = await session.create_session(
        body=session.CreateSessionBody(ownerType="t", ownerId="o"),
    )
    assert resp.sessionId == "existing-uuid"
    assert resp.state == "ready"
    mock_batch_v1.create_namespaced_job.assert_not_called()


async def test_create_session_marks_down_on_k8s_failure(manager_env, mock_batch_v1, mocker):
    mocker.patch("web_api.app.session._reserve_or_get_existing", return_value=None)
    write_state = mocker.patch("web_api.app.session._write_session_state")
    mock_batch_v1.create_namespaced_job.side_effect = ApiException(status=500)

    from web_api.app import session

    with pytest.raises(HTTPException) as exc:
        await session.create_session(
            body=session.CreateSessionBody(ownerType="t", ownerId="o"),
        )
    assert exc.value.status_code == 502
    assert write_state.call_args.kwargs["reason"] == "manager_job_create_failed"


async def test_dispatch_preparing_writes_queue(manager_env, mocker):
    mocker.patch("web_api.app.session._read_session_row", return_value={"state": "preparing"})
    write_queue = mocker.patch("web_api.app.session._write_queue_doc")

    from web_api.app import session

    resp = await session.dispatch(
        session_id="s1",
        body=session.DispatchBody(specUrl="gs://x", runId="r1", jobType="j"),
    )
    assert resp["state"] == "queued-pre-ready"
    write_queue.assert_called_once()


async def test_dispatch_concurrent_preparing_all_enqueued(manager_env, mocker):
    """N concurrent pre-ready dispatches each enqueue."""
    mocker.patch("web_api.app.session._read_session_row", return_value={"state": "preparing"})
    write_queue = mocker.patch("web_api.app.session._write_queue_doc")

    from web_api.app import session

    await asyncio.gather(
        *[
            session.dispatch(
                session_id="s1",
                body=session.DispatchBody(specUrl=f"gs://x{i}", runId=f"r{i}", jobType="j"),
            )
            for i in range(10)
        ]
    )
    assert write_queue.call_count == 10


async def test_dispatch_ready_proxies_to_master(manager_env, mocker, aiohttp_mock_session):
    mocker.patch(
        "web_api.app.session._read_session_row",
        return_value={"state": "ready", "controlEndpoint": "http://session-master-s1/"},
    )
    aiohttp_mock_session.set_post_response(
        status=200, json_payload={"result": 1, "dispatchCount": 1, "nearRecycle": False}
    )

    from web_api.app import session

    resp = await session.dispatch(
        session_id="s1",
        body=session.DispatchBody(specUrl="gs://x", runId="r1", jobType="j"),
    )
    assert resp["dispatchCount"] == 1


async def test_dispatch_ready_proxy_unreachable_lazy_down(
    manager_env, mocker, aiohttp_mock_session
):
    mocker.patch(
        "web_api.app.session._read_session_row",
        return_value={"state": "ready", "controlEndpoint": "http://nonexistent/"},
    )
    write_state = mocker.patch("web_api.app.session._write_session_state")
    aiohttp_mock_session.post.side_effect = aiohttp.ClientConnectionError("no route")

    from web_api.app import session

    with pytest.raises(HTTPException) as exc:
        await session.dispatch(
            session_id="s1",
            body=session.DispatchBody(specUrl="gs://x", runId="r1", jobType="j"),
        )
    assert exc.value.status_code == 502
    write_state.assert_called_with("s1", "down", reason="proxy_unreachable")


async def test_dispatch_ready_passes_through_master_error(
    manager_env, mocker, aiohttp_mock_session
):
    """A master HTTP error (e.g. 409 not-ready) surfaces with its own status,
    not a generic 500."""
    mocker.patch(
        "web_api.app.session._read_session_row",
        return_value={"state": "ready", "controlEndpoint": "http://session-master-s1/"},
    )
    aiohttp_mock_session.set_post_response(
        status=409, json_payload={"detail": "session state='preparing'"}
    )

    from web_api.app import session

    with pytest.raises(HTTPException) as exc:
        await session.dispatch(
            session_id="s1",
            body=session.DispatchBody(specUrl="gs://x", runId="r1", jobType="j"),
        )
    assert exc.value.status_code == 409
    assert exc.value.detail == "session state='preparing'"


async def test_status_preparing_returns_queue_depth(manager_env, mocker):
    mocker.patch("web_api.app.session._read_session_row", return_value={"state": "preparing"})
    mocker.patch("web_api.app.session._queue_depth", return_value=2)

    from web_api.app import session

    resp = await session.status(session_id="s1")
    assert resp == {"state": "preparing", "queueDepth": 2}


async def test_terminate_deletes_job_and_service_and_writes_down(
    manager_env, mock_batch_v1, mocker
):
    svc_delete = mocker.patch("web_api.app.session.service.delete_namespaced_service")
    write_state = mocker.patch("web_api.app.session._write_session_state")

    from web_api.app import session

    resp = await session.terminate(session_id="s1")
    assert resp == {"state": "down"}
    assert mock_batch_v1.delete_namespaced_job.call_args.kwargs["name"] == "session-master-s1"
    assert svc_delete.call_args.kwargs["name"] == "session-master-s1"
    write_state.assert_called_with("s1", "down", reason="explicit_terminate")


async def test_terminate_swallows_job_404(manager_env, mock_batch_v1, mocker):
    mocker.patch("web_api.app.session.service.delete_namespaced_service")
    mocker.patch("web_api.app.session._write_session_state")
    mock_batch_v1.delete_namespaced_job.side_effect = ApiException(status=404)

    from web_api.app import session

    resp = await session.terminate(session_id="gone")
    assert resp == {"state": "down"}
