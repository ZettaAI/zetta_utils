"""
Per-session master process.

Lifecycle:
  Boot     — read SESSION_ID, fetch Firestore row, create worker Pod+Service,
             poll /healthz, drain pre-ready queue, transition state=ready.
  Steady   — aiohttp.web app exposing /dispatch, /status, /terminate.
             Proxies to worker. Owns the cancellable idle TTL timer.
  Terminate — on SIGTERM (Job deletion) or idle-fire: delete worker, write
              state=down, exit cleanly.

Run via:  zetta session-master  (registered in cli/main.py)
"""

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import aiohttp
import yaml
from aiohttp import web
from google.cloud import firestore

from kubernetes import client as k8s_client
from zetta_utils.cloud_management.resource_allocation.k8s import pod, service
from zetta_utils.session import _get_sessions_db

log = logging.getLogger(__name__)


def _session_id() -> str:
    return os.environ["SESSION_ID"]


def _pod_name() -> str:
    return os.environ["POD_NAME"]


def _pod_uid() -> str:
    return os.environ["POD_UID"]


def _workload_namespace() -> str:
    return os.environ.get("WORKLOAD_NAMESPACE", "sessions")


def _sessions_image_tag() -> str:
    return os.environ["SESSIONS_IMAGE_TAG"]


def _worker_template_path() -> str:
    return os.environ["SESSION_WORKER_TEMPLATE_PATH"]


def _worker_service_template_path() -> str:
    return os.environ["SESSION_WORKER_SERVICE_TEMPLATE_PATH"]


WORKER_HEALTHZ_TIMEOUT_S = 60
WORKER_HEALTHZ_POLL_INTERVAL_S = 1
WORKER_HEALTHZ_REFUSAL_THRESHOLD = 5

# Master-local state — disposable; rebuilt by manager creating a new master
# if this process dies.
#
# Concurrency invariant: no asyncio.Lock is required around these globals.
# The worker enforces single-flight via a Semaphore(1) around its /run_spec/
# handler, so master can only have ONE response in flight at a time.
# _idle_timer_task is touched only from dispatch handlers (each serialised
# behind the worker semaphore anyway) and from the timer body itself; no race.
_idle_timer_task: asyncio.Task | None = None
_idle_ttl_sec: float = 3600
_worker_endpoint: str = ""
_shutdown_started: bool = False

_shutdown_event: asyncio.Event | None = None


def _get_shutdown_event() -> asyncio.Event:
    """Return the module-level shutdown event, creating it on first use.

    Bound to the loop running ``main()`` (the sole loop ``asyncio.run`` creates
    in the CLI entrypoint). ``_request_serve_stop`` and ``_serve_forever`` share
    it so a stop requested before serving begins is not lost.
    """
    global _shutdown_event  # pylint: disable=global-statement
    if _shutdown_event is None:
        _shutdown_event = asyncio.Event()
    return _shutdown_event


def _request_serve_stop() -> None:
    """Signal ``_serve_forever`` to return so the process can exit cleanly.

    Sets the shutdown event ``_serve_forever`` awaits. Touches no event-loop
    control directly, so callers (idle timer, terminate handler, SIGTERM
    handler) leave the loop free for in-flight work to drain.
    """
    _get_shutdown_event().set()


# ---- Firestore helpers --------------------------------------------------


def _read_session_row(session_id: str) -> dict:
    """Read the ``sessions/<session_id>`` document. Returns ``{}`` if absent."""
    snapshot = _get_sessions_db().collection("sessions").document(session_id).get()
    return snapshot.to_dict() or {}  # type: ignore[union-attr]


def _write_session_state(state: str, *, reason: str | None = None) -> None:
    """Merge ``state`` onto ``sessions/<SESSION_ID>``.

    When transitioning to ``down``, also stamps ``terminatedAt`` and
    ``terminationReason``.
    """
    payload: dict = {"state": state}
    if state == "down":
        payload["terminatedAt"] = datetime.now(timezone.utc)
        if reason is not None:
            payload["terminationReason"] = reason
    _get_sessions_db().collection("sessions").document(_session_id()).set(payload, merge=True)


def _read_queue_docs(session_id: str) -> list[dict]:
    """List ``sessions/<session_id>/queue/*`` ordered by ``enqueuedAt`` asc.

    Each returned dict carries the document fields plus its ``dispatchId``
    (the document id).
    """
    query = (
        _get_sessions_db()
        .collection("sessions")
        .document(session_id)
        .collection("queue")
        .order_by("enqueuedAt", direction=firestore.Query.ASCENDING)
    )
    out: list[dict] = []
    for snapshot in query.stream():
        doc = snapshot.to_dict() or {}
        doc["dispatchId"] = snapshot.id
        out.append(doc)
    return out


def _delete_queue_doc(session_id: str, dispatch_id: str) -> None:
    """Delete ``sessions/<session_id>/queue/<dispatch_id>``."""
    (
        _get_sessions_db()
        .collection("sessions")
        .document(session_id)
        .collection("queue")
        .document(dispatch_id)
        .delete()
    )


def _update_last_dispatch_at() -> None:
    """Merge ``lastDispatchAt=<server timestamp>`` onto ``sessions/<SESSION_ID>``."""
    _get_sessions_db().collection("sessions").document(_session_id()).set(
        {"lastDispatchAt": firestore.SERVER_TIMESTAMP}, merge=True
    )


# ---- Boot ---------------------------------------------------------------


async def main() -> None:
    """CLI entrypoint. Boots master, serves the app, blocks until exit signal."""
    _install_sigterm_handler()
    try:
        await _boot()
        await _serve_forever()
    finally:
        await _on_shutdown(reason="explicit_terminate")


async def _boot() -> None:
    global _worker_endpoint, _idle_ttl_sec  # pylint: disable=global-statement

    session_id = _session_id()
    namespace = _workload_namespace()
    log.info("sessions.master.boot_start", extra={"sessionId": session_id})
    row = _read_session_row(session_id)
    if row.get("state") != "preparing":
        log.error(
            "sessions.master.unexpected_initial_state",
            extra={"sessionId": session_id, "state": row.get("state")},
        )
        raise SystemExit(2)

    _idle_ttl_sec = int(row.get("config", {}).get("idleTtlSec", 3600))
    initial_preload = row.get("initialPreload", "try")

    # Render the worker template; create Pod + Service with ownerReferences
    # pointing at THIS master Pod (downward-API env).
    worker_body = _render_worker_template(initial_preload=initial_preload)
    pod.create_namespaced_pod(namespace=namespace, body=worker_body)

    worker_svc_body = _render_worker_service()
    service.create_namespaced_service(namespace=namespace, body=worker_svc_body)

    _worker_endpoint = f"http://session-worker-{session_id}.{namespace}.svc.cluster.local/"

    await _wait_for_worker_healthz()
    await _drain_pre_ready_queue()
    _write_session_state("ready")
    log.info("sessions.master.boot_complete", extra={"sessionId": session_id})
    _start_idle_timer()


def _render_worker_template(*, initial_preload: str) -> dict:
    """Load the worker Pod YAML template and substitute placeholders."""
    raw = Path(_worker_template_path()).read_text(encoding="utf-8")
    substituted = (
        raw.replace("${SESSION_ID}", _session_id())
        .replace("${INITIAL_PRELOAD}", initial_preload)
        .replace("${MASTER_POD_NAME}", _pod_name())
        .replace("${MASTER_POD_UID}", _pod_uid())
        .replace("${SESSIONS_IMAGE_TAG}", _sessions_image_tag())
    )
    return yaml.safe_load(substituted)


def _render_worker_service() -> dict:
    """Load the worker Service YAML template and substitute placeholders."""
    raw = Path(_worker_service_template_path()).read_text(encoding="utf-8")
    substituted = (
        raw.replace("${SESSION_ID}", _session_id())
        .replace("${MASTER_POD_NAME}", _pod_name())
        .replace("${MASTER_POD_UID}", _pod_uid())
    )
    return yaml.safe_load(substituted)


# ---- Worker probing -----------------------------------------------------


async def _wait_for_worker_healthz() -> None:
    """Poll the worker ``/healthz`` until ready or the boot budget expires.

    Applies the failure-mode disambiguation predicate on repeated
    connection-refused; logs a terminal verdict and terminates the session on
    a permanent worker failure or on timeout.
    """
    refusal_count = 0
    deadline = asyncio.get_event_loop().time() + WORKER_HEALTHZ_TIMEOUT_S
    timeout = aiohttp.ClientTimeout(total=2.0)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            now = asyncio.get_event_loop().time()
            if now >= deadline:
                await _terminate_session("worker_healthz_timeout")
            try:
                async with session.get(f"{_worker_endpoint}healthz") as response:
                    if response.status == 200:
                        return
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError):
                refusal_count += 1
                if refusal_count >= WORKER_HEALTHZ_REFUSAL_THRESHOLD:
                    verdict = _classify_worker_failure()
                    if verdict == "permanent":
                        await _terminate_session("worker_permanent_failure")
                    # else: continue polling within the 60s budget
            await asyncio.sleep(WORKER_HEALTHZ_POLL_INTERVAL_S)


def _classify_worker_failure() -> Literal["permanent", "transient"]:
    """Classify a worker outage from its Pod status.

    ``phase=='Failed'`` or a non-zero container exit code is ``"permanent"``;
    ``phase`` of ``Pending`` / ``Running`` is ``"transient"`` (keep polling
    within the boot budget); unknown / ``Succeeded`` is ``"permanent"`` (the
    worker is gone).
    """
    session_id = _session_id()
    try:
        worker = pod.read_namespaced_pod_status(
            name=f"session-worker-{session_id}",
            namespace=_workload_namespace(),
        )
    except k8s_client.exceptions.ApiException as e:
        if e.status == 404:
            log.warning(
                "sessions.master.worker_404",
                extra={"sessionId": session_id, "context": "classify"},
            )
            return "permanent"
        raise

    phase = worker.status.phase
    if phase == "Failed":
        exit_code = None
        for cs in worker.status.container_statuses or []:
            term = getattr(cs.state, "terminated", None)
            if term and term.exit_code is not None:
                exit_code = term.exit_code
                break
        log.error(
            "sessions.worker.permanent_failure",
            extra={"sessionId": session_id, "exitCode": exit_code},
        )
        return "permanent"

    if phase in ("Pending", "Running"):
        return "transient"

    # Unknown / Succeeded — treat as permanent (worker is gone).
    return "permanent"


# ---- Queue drain --------------------------------------------------------


async def _drain_pre_ready_queue() -> None:
    """Drain the pre-ready queue, forwarding each dispatch to the worker.

    Polls until the queue is empty. Firestore reads are strongly consistent
    within a region and the manager writes queue rows before transitioning the
    session row to ``preparing``, so all enqueued rows are visible. Any row
    enqueued after the drain completes is handled by the regular ``/dispatch``
    path in the ready state.
    """
    session_id = _session_id()
    drained = 0
    while True:
        docs = _read_queue_docs(session_id)  # ordered by enqueuedAt asc
        if not docs:
            break
        for doc in docs:
            await _forward_dispatch_to_worker(doc)
            _delete_queue_doc(session_id, doc["dispatchId"])
            drained += 1
    log.info(
        "sessions.master.queue_drained",
        extra={"sessionId": session_id, "drainedCount": drained},
    )


# ---- Endpoint logic -----------------------------------------------------


async def _dispatch_logic(body: dict, *, authorization: str | None) -> dict:
    """Forward a dispatch to the worker, pausing the idle timer around it."""
    _cancel_idle_timer()
    try:
        return await _forward_dispatch_to_worker(body, user_token=authorization)
    finally:
        _start_idle_timer()


async def _status_logic() -> dict:
    """Probe the worker ``/healthz`` and report the session state.

    On an unreachable worker, mark the session ``down`` (reason
    ``proxy_unreachable``) and return ``{"state": "down"}``.
    """
    try:
        timeout = aiohttp.ClientTimeout(total=5.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{_worker_endpoint}healthz") as response:
                return {"state": "ready" if response.status == 200 else "down"}
    except (aiohttp.ClientConnectionError, asyncio.TimeoutError):
        log.warning(
            "sessions.master.worker_404",
            extra={"sessionId": _session_id(), "context": "status"},
        )
        _write_session_state("down", reason="proxy_unreachable")
        return {"state": "down"}


async def _terminate_logic() -> dict:
    """Run a clean shutdown then request the serve loop to stop."""
    await _on_shutdown(reason="explicit_terminate")
    _request_serve_stop()
    return {"state": "down"}


# ---- aiohttp.web handlers -----------------------------------------------


async def dispatch(request: web.Request) -> web.Response:
    body = await request.json()
    payload = {
        "specUrl": body.get("specUrl"),
        "runId": body.get("runId"),
        "jobType": body.get("jobType"),
        "requiredPreload": body.get("requiredPreload", "try"),
    }
    authorization = request.headers.get("Authorization")
    result = await _dispatch_logic(payload, authorization=authorization)
    return web.json_response(result)


async def status(request: web.Request) -> web.Response:  # pylint: disable=unused-argument
    return web.json_response(await _status_logic())


async def terminate(request: web.Request) -> web.Response:  # pylint: disable=unused-argument
    return web.json_response(await _terminate_logic())


def _build_app() -> web.Application:
    app = web.Application()
    app.router.add_post("/dispatch", dispatch)
    app.router.add_get("/status", status)
    app.router.add_post("/terminate", terminate)
    return app


# ---- Forwarding to worker -----------------------------------------------


async def _forward_dispatch_to_worker(
    dispatch_doc: dict,
    *,
    user_token: str | None = None,
) -> dict:
    headers = {}
    token = user_token
    if token:
        if not token.startswith("Bearer "):
            token = "Bearer " + token
        headers["Authorization"] = token

    payload = {
        k: dispatch_doc[k]
        for k in ("specUrl", "runId", "jobType", "requiredPreload")
        if k in dispatch_doc
    }

    try:
        # timeout=None lets the worker take as long as it needs (mazepa builds
        # can run minutes). allow_redirects=False matches the worker's
        # redirect_slashes=False config — a redirect would indicate drift.
        timeout = aiohttp.ClientTimeout(total=None)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{_worker_endpoint}run_spec/",
                json=payload,
                headers=headers,
                allow_redirects=False,
            ) as response:
                response.raise_for_status()
                body = await response.json()
        _update_last_dispatch_at()
        return body
    except aiohttp.ClientResponseError as e:
        # Worker process is intact (no connection refusal) but /run_spec/
        # returned >= 400. 4xx never retries or recycles. 5xx gets one bounded
        # retry then surfaces 502. No recycle — the worker is healthy at the
        # process level; the failure is application-level.
        if e.status < 500:
            raise web.HTTPBadGateway(reason="worker_run_spec_client_error") from e
        if dispatch_doc.get("_master_retry_attempted"):
            raise web.HTTPBadGateway(reason="worker_run_spec_error") from e
        await asyncio.sleep(2.0)
        dispatch_doc = {**dispatch_doc, "_master_retry_attempted": True}
        return await _forward_dispatch_to_worker(dispatch_doc, user_token=user_token)
    except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
        # Worker is unreachable. Classify and either keep going or terminate.
        verdict = _classify_worker_failure()
        if verdict == "permanent":
            await _terminate_session("worker_permanent_failure")
        raise web.HTTPBadGateway(reason="worker unreachable") from e


# ---- Idle timer ---------------------------------------------------------


def _start_idle_timer() -> None:
    global _idle_timer_task  # pylint: disable=global-statement
    if _idle_timer_task and not _idle_timer_task.done():
        return
    _idle_timer_task = asyncio.create_task(_idle_timer_body())


def _cancel_idle_timer() -> None:
    global _idle_timer_task  # pylint: disable=global-statement
    if _idle_timer_task and not _idle_timer_task.done():
        _idle_timer_task.cancel()
    _idle_timer_task = None


async def _idle_timer_body() -> None:
    try:
        await asyncio.sleep(_idle_ttl_sec)
    except asyncio.CancelledError:
        return
    log.info(
        "sessions.master.idle_timer_fired",
        extra={"sessionId": _session_id(), "idleTtlSec": _idle_ttl_sec},
    )
    await _on_shutdown(reason="idle_timer")
    _request_serve_stop()


# ---- Shutdown -----------------------------------------------------------


def _install_sigterm_handler() -> None:  # pragma: no cover
    """Install a SIGTERM handler that runs a clean shutdown.

    Uses the running loop's ``add_signal_handler`` so the callback runs from
    inside the loop (it integrates with the loop's selector). Falls back to
    ``signal.signal`` only on platforms that do not support the loop API
    (non-POSIX dev environments); production runs on Linux pods.
    """

    def _on_sigterm() -> None:
        asyncio.create_task(_on_shutdown(reason="explicit_terminate"))
        _request_serve_stop()

    try:
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, _on_sigterm)
    except NotImplementedError:
        signal.signal(signal.SIGTERM, lambda *_: _on_sigterm())


async def _on_shutdown(*, reason: str) -> None:
    global _shutdown_started  # pylint: disable=global-statement
    if _shutdown_started:
        return
    _shutdown_started = True
    session_id = _session_id()
    namespace = _workload_namespace()
    _write_session_state("down", reason=reason)
    pod.delete_namespaced_pod(
        name=f"session-worker-{session_id}",
        namespace=namespace,
    )
    service.delete_namespaced_service(
        name=f"session-worker-{session_id}",
        namespace=namespace,
    )


async def _terminate_session(reason: str) -> None:
    await _on_shutdown(reason=reason)
    sys.exit(1 if reason != "idle_timer" else 0)


# ---- Serve --------------------------------------------------------------


async def _serve_forever() -> None:  # pragma: no cover
    """Run the aiohttp.web app, bound to 0.0.0.0:80.

    Returns when the shutdown event is set by the idle timer, the terminate
    handler, or the SIGTERM handler, allowing the process to exit cleanly.
    """
    runner = web.AppRunner(_build_app())
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 80)
    await site.start()
    await _get_shutdown_event().wait()
    await runner.cleanup()
