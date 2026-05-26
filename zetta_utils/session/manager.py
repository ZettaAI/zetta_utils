# pylint: disable=all # type: ignore
"""
Always-on session-manager service.

Endpoints:
  POST   /sessions
  POST   /sessions/{id}/dispatch
  GET    /sessions/{id}/status
  DELETE /sessions/{id}

Auth: forwards the caller's Bearer token to the master for worker-side use.
Access control is enforced by the NetworkPolicy at the cluster boundary.

State: 100% Firestore-backed (main DB). No in-memory session list, no reaper.
HTTP proxying uses aiohttp (already a dependency; mirrors the master).
"""

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import aiohttp
import hypercorn
import hypercorn.asyncio
import yaml
from fastapi import FastAPI, Header, HTTPException
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from pydantic import BaseModel

from kubernetes import client as k8s_client
from zetta_utils.cloud_management.resource_allocation.k8s import service
from zetta_utils.session import _get_sessions_db

log = logging.getLogger(__name__)

WORKLOAD_NAMESPACE = os.environ.get("WORKLOAD_NAMESPACE", "sessions")
SESSIONS_IMAGE_TAG = os.environ["SESSIONS_IMAGE_TAG"]
MASTER_TEMPLATE_PATH = os.environ["SESSION_MASTER_TEMPLATE_PATH"]
MASTER_SERVICE_TEMPLATE_PATH = os.environ["SESSION_MASTER_SERVICE_TEMPLATE_PATH"]
ACTIVE_SESSIONS_GAUGE_INTERVAL_S = 30


class CreateSessionBody(BaseModel):
    ownerType: str
    ownerId: str
    initialPreload: Literal["none", "try", "inference", "training", "all"] = "try"
    jobType: str | None = None
    config: dict | None = None


class CreateSessionResponse(BaseModel):
    sessionId: str
    controlEndpoint: str
    workerEndpoint: str
    state: Literal["preparing", "ready"]


class DispatchBody(BaseModel):
    specUrl: str
    runId: str
    jobType: str
    requiredPreload: Literal["none", "try", "inference", "training", "all"] = "try"


# ---- Firestore helpers --------------------------------------------------


def _read_session_row(session_id: str) -> dict | None:
    """Read the ``sessions/<session_id>`` document; ``None`` if absent."""
    snap = _get_sessions_db().collection("sessions").document(session_id).get()
    return snap.to_dict() if snap.exists else None


def _write_session_state(session_id: str, state: str, *, reason: str | None = None) -> None:
    """Merge ``state`` onto ``sessions/<session_id>``.

    When transitioning to ``down``, also stamps ``terminatedAt`` and
    ``terminationReason``.
    """
    payload: dict = {"state": state}
    if state == "down":
        payload["terminatedAt"] = firestore.SERVER_TIMESTAMP
        if reason is not None:
            payload["terminationReason"] = reason
    _get_sessions_db().collection("sessions").document(session_id).set(payload, merge=True)


def _write_queue_doc(session_id: str, dispatch_id: str, fields: dict) -> None:
    """Write ``sessions/<session_id>/queue/<dispatch_id>``."""
    (
        _get_sessions_db()
        .collection("sessions")
        .document(session_id)
        .collection("queue")
        .document(dispatch_id)
        .set(fields)
    )


def _queue_depth(session_id: str) -> int:
    """Count documents under ``sessions/<session_id>/queue``."""
    docs = (
        _get_sessions_db().collection("sessions").document(session_id).collection("queue").stream()
    )
    return sum(1 for _ in docs)


def _reserve_or_get_existing(
    owner_type: str, owner_id: str, session_id: str, new_row: dict
) -> dict | None:
    """Atomically reuse-or-reserve a session for an owner.

    In one Firestore transaction: if an active session (state in
    ``{preparing, ready}``) for the owner already exists, return it (carrying
    its ``sessionId`` and ``state``); otherwise write ``new_row`` at
    ``sessions/<session_id>`` and return ``None``.

    Guards the lookup-then-create race so two concurrent ``POST /sessions`` for
    the same owner cannot both spawn a master. Firestore requires all reads
    before any write inside a transaction, and writes go through the
    transaction object. Needs the composite index
    ``(ownerType, ownerId, state, createdAt DESC)``.
    """
    db = _get_sessions_db()
    sessions = db.collection("sessions")
    transaction = db.transaction()

    @firestore.transactional
    def _run(txn) -> dict | None:
        query = (
            sessions.where(filter=FieldFilter("ownerType", "==", owner_type))
            .where(filter=FieldFilter("ownerId", "==", owner_id))
            .where(filter=FieldFilter("state", "in", ["preparing", "ready"]))
            .order_by("createdAt", direction=firestore.Query.DESCENDING)
            .limit(1)
        )
        found = list(query.stream(transaction=txn))
        if found:
            row = found[0].to_dict() or {}
            row["sessionId"] = found[0].id
            return row
        txn.set(sessions.document(session_id), new_row)
        return None

    return _run(transaction)


# ---- Auth + proxy helpers -----------------------------------------------


async def _safe_detail(response: aiohttp.ClientResponse) -> str:
    """Best-effort extract the master's error ``detail``.

    Reads the FastAPI ``HTTPException`` JSON shape; falls back to the HTTP
    reason phrase when the body is not JSON.
    """
    try:
        return (await response.json()).get("detail", response.reason)
    except Exception:  # pylint: disable=broad-exception-caught
        return response.reason or "session master error"


# ---- K8s rendering ------------------------------------------------------


def _build_endpoints(session_id: str) -> tuple[str, str]:
    """Return the ``(controlEndpoint, workerEndpoint)`` cluster DNS URLs."""
    control = f"http://session-master-{session_id}.{WORKLOAD_NAMESPACE}.svc.cluster.local/"
    worker = f"http://session-worker-{session_id}.{WORKLOAD_NAMESPACE}.svc.cluster.local/"
    return control, worker


def _render_master_job(*, session_id: str) -> dict:
    """Load the master Job YAML template and substitute placeholders."""
    raw = Path(MASTER_TEMPLATE_PATH).read_text()
    substituted = raw.replace("${SESSION_ID}", session_id).replace(
        "${SESSIONS_IMAGE_TAG}", SESSIONS_IMAGE_TAG
    )
    return yaml.safe_load(substituted)


def _render_master_service(*, session_id: str, job_uid: str) -> dict:
    """Load the master Service YAML template and substitute placeholders."""
    raw = Path(MASTER_SERVICE_TEMPLATE_PATH).read_text()
    substituted = raw.replace("${SESSION_ID}", session_id).replace("${MASTER_JOB_UID}", job_uid)
    return yaml.safe_load(substituted)


# ---- Endpoints ----------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):
    gauge = asyncio.create_task(_active_sessions_gauge_loop())
    try:
        yield
    finally:
        gauge.cancel()


app = FastAPI(lifespan=_lifespan)


@app.post("/sessions", response_model=CreateSessionResponse, status_code=201)
async def create_session(
    body: CreateSessionBody,
    authorization: str = Header(...),
) -> CreateSessionResponse:
    session_id = str(uuid.uuid4())
    control, worker_url = _build_endpoints(session_id)

    config = dict(body.config or {})
    config.setdefault("idleTtlSec", 3600)
    config.setdefault("maxDispatches", 50)

    new_row = {
        "ownerType": body.ownerType,
        "ownerId": body.ownerId,
        "state": "preparing",
        "controlEndpoint": control,
        "workerEndpoint": worker_url,
        "initialPreload": body.initialPreload,
        "jobType": body.jobType,
        "config": config,
        "createdAt": firestore.SERVER_TIMESTAMP,
    }

    # Atomic reuse-or-reserve closes the concurrent-create race: either we
    # reserve session_id (returns None) or we get back a pre-existing/raced
    # active session and skip Job creation entirely.
    existing = _reserve_or_get_existing(body.ownerType, body.ownerId, session_id, new_row)
    if existing is not None:
        ctrl, wkr = _build_endpoints(existing["sessionId"])
        log.info(
            "sessions.session.reused",
            extra={
                "sessionId": existing["sessionId"],
                "ownerType": body.ownerType,
                "ownerId": body.ownerId,
            },
        )
        return CreateSessionResponse(
            sessionId=existing["sessionId"],
            controlEndpoint=ctrl,
            workerEndpoint=wkr,
            state=existing["state"],
        )

    try:
        job = k8s_client.BatchV1Api().create_namespaced_job(
            namespace=WORKLOAD_NAMESPACE,
            body=_render_master_job(session_id=session_id),
        )
        # The master Service is what makes controlEndpoint resolve.
        # ownerReferences point at the Job, so it is cascade-GC'd when the Job
        # is deleted (terminate) or TTL-reaped after the master exits.
        service.create_namespaced_service(
            namespace=WORKLOAD_NAMESPACE,
            body=_render_master_service(session_id=session_id, job_uid=job.metadata.uid),
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        _write_session_state(session_id, "down", reason="manager_job_create_failed")
        log.error(
            "sessions.manager.session_create_failed",
            extra={
                "ownerType": body.ownerType,
                "ownerId": body.ownerId,
                "error": str(e),
            },
        )
        raise HTTPException(status_code=502, detail="manager could not create master Job/Service")

    log.info(
        "sessions.session.created",
        extra={
            "sessionId": session_id,
            "ownerType": body.ownerType,
            "jobType": body.jobType,
        },
    )
    return CreateSessionResponse(
        sessionId=session_id,
        controlEndpoint=control,
        workerEndpoint=worker_url,
        state="preparing",
    )


@app.post("/sessions/{session_id}/dispatch")
async def dispatch(
    session_id: str,
    body: DispatchBody,
    authorization: str = Header(...),
) -> dict:
    row = _read_session_row(session_id)
    if row is None:
        raise HTTPException(status_code=404, detail="unknown session")

    if row["state"] == "preparing":
        dispatch_id = str(uuid.uuid4())
        _write_queue_doc(
            session_id,
            dispatch_id,
            {
                "specUrl": body.specUrl,
                "runId": body.runId,
                "jobType": body.jobType,
                "requiredPreload": body.requiredPreload,
                "userToken": authorization,
                "enqueuedAt": firestore.SERVER_TIMESTAMP,
            },
        )
        return {"runId": body.runId, "state": "queued-pre-ready"}

    if row["state"] == "ready":
        try:
            timeout = aiohttp.ClientTimeout(total=None)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{row['controlEndpoint']}dispatch",
                    json=body.model_dump(),
                    headers={"Authorization": authorization},
                ) as response:
                    if response.status >= 400:
                        # Surface the master's status + detail (401/409/502...)
                        # instead of a generic 500. HTTPException propagates
                        # past the connection-error handler below untouched.
                        raise HTTPException(
                            status_code=response.status,
                            detail=await _safe_detail(response),
                        )
                    return await response.json()
        except (aiohttp.ClientConnectionError, asyncio.TimeoutError):
            _write_session_state(session_id, "down", reason="proxy_unreachable")
            log.warning(
                "sessions.manager.proxy_unreachable",
                extra={"sessionId": session_id, "endpoint": row["controlEndpoint"]},
            )
            raise HTTPException(status_code=502, detail="session master unreachable")

    raise HTTPException(status_code=409, detail=f"session state={row['state']!r}")


@app.get("/sessions/{session_id}/status")
async def status(
    session_id: str,
    authorization: str = Header(...),
) -> dict:
    row = _read_session_row(session_id)
    if row is None:
        raise HTTPException(status_code=404)

    if row["state"] == "preparing":
        return {"state": "preparing", "queueDepth": _queue_depth(session_id)}

    try:
        timeout = aiohttp.ClientTimeout(total=5.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(
                f"{row['controlEndpoint']}status",
                headers={"Authorization": authorization},
            ) as response:
                if response.status >= 400:
                    raise HTTPException(
                        status_code=response.status,
                        detail=await _safe_detail(response),
                    )
                return await response.json()
    except (aiohttp.ClientConnectionError, asyncio.TimeoutError):
        _write_session_state(session_id, "down", reason="proxy_unreachable")
        log.warning(
            "sessions.manager.proxy_unreachable",
            extra={"sessionId": session_id, "endpoint": row["controlEndpoint"]},
        )
        return {"state": "down"}


@app.delete("/sessions/{session_id}")
async def terminate(
    session_id: str,
    authorization: str = Header(...),
) -> dict:
    try:
        k8s_client.BatchV1Api().delete_namespaced_job(
            name=f"session-master-{session_id}",
            namespace=WORKLOAD_NAMESPACE,
            propagation_policy="Background",
        )
    except k8s_client.exceptions.ApiException as e:
        if e.status not in (404, 410):
            raise
    # ownerReferences point the master Service at the Job for cascade GC, but
    # delete it explicitly for promptness (swallows 404/410).
    service.delete_namespaced_service(
        name=f"session-master-{session_id}", namespace=WORKLOAD_NAMESPACE
    )
    _write_session_state(session_id, "down", reason="explicit_terminate")
    return {"state": "down"}


async def _active_sessions_gauge_loop() -> None:
    while True:
        try:
            jobs = k8s_client.BatchV1Api().list_namespaced_job(
                namespace=WORKLOAD_NAMESPACE,
                label_selector="app=session-master",
            )
            count = sum(1 for j in jobs.items if j.status.active)
            log.info("sessions.active_sessions_count", extra={"value": count})
        except Exception as e:  # pylint: disable=broad-exception-caught
            log.warning(
                "sessions.active_sessions_count_sample_failed",
                extra={"error": str(e)},
            )
        await asyncio.sleep(ACTIVE_SESSIONS_GAUGE_INTERVAL_S)


async def main() -> None:
    """CLI entrypoint. Serve the FastAPI app via hypercorn (mirrors master)."""
    config = hypercorn.Config()
    config.bind = ["0.0.0.0:80"]
    await hypercorn.asyncio.serve(app, config)
