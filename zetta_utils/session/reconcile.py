"""
Daily reconcile backstop.

Predicates per session row:
  (1) state != "down" AND lastDispatchAt < now() - 24h
   OR
  (2) state != "down" AND BatchV1Api.read_namespaced_job(name=session-master-<id>)
      raises ApiException(status in (404, 410))

On match: write state="down" + terminatedAt + terminationReason, then clean up
based on which predicate fired:
  - master MISSING (Job already 404/410): the orphan case cascade-GC missed.
    Best-effort delete the worker Pod, worker Service, and master Service.
  - stale-but-ALIVE (Job still exists): delete the master Job (Background
    propagation); ownerReferences cascade-GC the worker Pod/Service + master
    Service. This actually terminates the still-running master.
Swallow 404/410 on every delete; count other errors as cleanupErrors.
"""

import logging
import os
from datetime import datetime, timedelta, timezone

from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

from kubernetes import client as k8s_client
from zetta_utils.cloud_management.resource_allocation.k8s import pod, service
from zetta_utils.session import _get_sessions_db

log = logging.getLogger(__name__)

WORKLOAD_NAMESPACE = os.environ.get("WORKLOAD_NAMESPACE", "sessions")
STALE_AFTER = timedelta(hours=24)


def run_reconcile() -> dict:
    total = 0
    stale_by_time = 0
    stale_by_missing_master = 0
    reconciled = 0
    cleanup_errors = 0

    batch_v1 = k8s_client.BatchV1Api()
    now = datetime.now(timezone.utc)

    for row in _query_non_down_sessions():
        total += 1
        session_id = row["sessionId"]
        is_stale_by_time = _is_stale_by_time(row, now)
        is_missing_master = _is_master_missing(batch_v1, session_id)

        if not (is_stale_by_time or is_missing_master):
            continue

        # Missing-master takes precedence: it dictates orphan cleanup and is the
        # more actionable signal when a row is both stale and missing.
        if is_missing_master:
            stale_by_missing_master += 1
            reason = "reconcile_master_missing"
        else:
            stale_by_time += 1
            reason = "reconcile_stale_24h"

        _write_session_state(session_id, "down", reason=reason)

        try:
            if is_missing_master:
                pod.delete_namespaced_pod(
                    name=f"session-worker-{session_id}", namespace=WORKLOAD_NAMESPACE
                )
                service.delete_namespaced_service(
                    name=f"session-worker-{session_id}", namespace=WORKLOAD_NAMESPACE
                )
                service.delete_namespaced_service(
                    name=f"session-master-{session_id}", namespace=WORKLOAD_NAMESPACE
                )
            else:
                # Stale but alive: delete the master Job; ownerReferences
                # cascade-GC the worker Pod/Service and master Service.
                _delete_master_job(batch_v1, session_id)
        except Exception as e:  # pylint: disable=broad-exception-caught
            cleanup_errors += 1
            log.warning(
                "sessions.reconcile.cleanup_error",
                extra={"sessionId": session_id, "error": str(e)},
            )

        log.info(
            "sessions.reconcile.reconciled_count",
            extra={"sessionId": session_id, "reason": reason},
        )
        log.info("sessions.session.terminated", extra={"sessionId": session_id, "reason": reason})
        reconciled += 1

    summary = {
        "totalRowsScanned": total,
        "staleByTime": stale_by_time,
        "staleByMissingMaster": stale_by_missing_master,
        "reconciledCount": reconciled,
        "cleanupErrors": cleanup_errors,
    }
    log.info("sessions.reconcile.scan_complete", extra=summary)
    return summary


def _is_stale_by_time(row: dict, now: datetime) -> bool:
    last = row.get("lastDispatchAt") or row.get("createdAt")
    if last is None:
        return False
    return (now - last) > STALE_AFTER


def _is_master_missing(batch_v1: k8s_client.BatchV1Api, session_id: str) -> bool:
    try:
        batch_v1.read_namespaced_job(
            name=f"session-master-{session_id}", namespace=WORKLOAD_NAMESPACE
        )
        return False
    except k8s_client.exceptions.ApiException as e:
        if e.status in (404, 410):
            return True
        raise


def _delete_master_job(batch_v1: k8s_client.BatchV1Api, session_id: str) -> None:
    """Delete the master Job (Background propagation; cascade-GC reaps the
    worker Pod/Service + master Service). Swallows 404/410 (already gone)."""
    try:
        batch_v1.delete_namespaced_job(
            name=f"session-master-{session_id}",
            namespace=WORKLOAD_NAMESPACE,
            propagation_policy="Background",
        )
    except k8s_client.exceptions.ApiException as e:
        if e.status not in (404, 410):
            raise


def _query_non_down_sessions():
    """Single-field query: yield rows where state != "down" (equivalently
    state in ("preparing", "ready", "working", "idle")), each carrying its
    document id as row["sessionId"]. Staleness is judged client-side from
    lastDispatchAt, so this needs no composite index."""
    query = (
        _get_sessions_db().collection("sessions").where(filter=FieldFilter("state", "!=", "down"))
    )
    for snap in query.stream():
        row = snap.to_dict() or {}
        row["sessionId"] = snap.id
        yield row


def _write_session_state(session_id: str, state: str, *, reason: str) -> None:
    """Merge-write to sessions/<id>; auto-stamps terminatedAt on 'down'
    (same convention as manager/master)."""
    payload = {"state": state}
    if state == "down":
        payload["terminatedAt"] = firestore.SERVER_TIMESTAMP
        payload["terminationReason"] = reason
    _get_sessions_db().collection("sessions").document(session_id).set(payload, merge=True)
