"""GC orchestrator: stale-run discovery, per-run cleanup, summary reporting.

Reads stale runs from ``RUN_DB``, groups each run's resources by cluster
via :mod:`discovery`, dispatches deletes through :mod:`deleters`, persists
per-run state in :mod:`state`, and emits a single multi-line summary log
per run (no per-resource log spam).
"""

from __future__ import annotations

import json
import os
import time
import traceback
from collections import defaultdict
from typing import Any

from google.api_core.exceptions import GoogleAPICallError

from zetta_utils.cloud_management.resource_allocation.k8s.common import ClusterInfo
from zetta_utils.log import get_logger
from zetta_utils.mazepa_addons.configurations.execute_on_gcp_with_sqs import (
    DEFAULT_GCP_CLUSTER,
)
from zetta_utils.run import RunInfo, RunState, update_run_info
from zetta_utils.run.db import RUN_DB
from zetta_utils.run.gc.common import CleanupReport, ResourceOutcome
from zetta_utils.run.gc.deleters import (
    K8S_DELETERS,
    SQS_DELETERS,
    DeleteOutcome,
    DeleteStatus,
    build_k8s_context,
    build_sqs_context,
)
from zetta_utils.run.gc.discovery import ClusterFailure, discover_locations
from zetta_utils.run.gc.slack import post_cycle, post_idle, post_message
from zetta_utils.run.gc.state import RunGCState, load_states, save_state
from zetta_utils.run.gc.users import UserResolver
from zetta_utils.run.gc.utils import (
    format_cluster,
    format_duration,
    purge_run_state,
    retried,
)
from zetta_utils.run.resource import RESOURCE_DB, Resource, deregister_resource

logger = get_logger("zetta_utils")

#: Priority order used to pick the dominant ``error_class`` for a run's
#: :class:`CleanupReport`. Cluster-wide failures outrank per-resource ones;
#: auth issues outrank transient 5xx; sqs / k8s_other are last resorts.
_ERROR_CLASS_PRIORITY = (
    "gc_internal",
    "cluster_404",
    "cluster_auth",
    "k8s_auth",
    "k8s_5xx",
    "sqs",
    "unknown_type",
    "firestore",
    "k8s_other",
)


def _parse_clusters(clusters_value: object) -> list[ClusterInfo]:
    if not isinstance(clusters_value, str) or not clusters_value:
        return [DEFAULT_GCP_CLUSTER]
    try:
        cluster_dicts = json.loads(clusters_value)
    except json.JSONDecodeError:
        return [DEFAULT_GCP_CLUSTER]
    return [ClusterInfo(**d) for d in cluster_dicts]


def _dominant_error_class(
    outcomes: list[ResourceOutcome],
    cluster_failures: dict[ClusterInfo, ClusterFailure],
) -> str:
    candidates: set[str] = set()
    for failure in cluster_failures.values():
        candidates.add(failure.error_class)
    for outcome in outcomes:
        if outcome.outcome.status == DeleteStatus.FAILED:
            candidates.add(outcome.outcome.error_class)
    for klass in _ERROR_CLASS_PRIORITY:
        if klass in candidates:
            return klass
    return ""


def _safe_deregister(resource_id: str) -> bool:
    """Deregister a resource with transient-API retries.

    Returns True on success, False on persistent failure (logged with the
    resource id and exception detail). A False result is converted into a
    ``FAILED("firestore")`` outcome by :func:`_apply_deregisters` so the
    run's status reflects the incomplete cleanup.
    """
    try:
        retried(lambda: deregister_resource(resource_id))
        return True
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning(f"Failed to deregister resource {resource_id}: {exc}")
        return False


def _apply_deregisters(outcomes: list[ResourceOutcome]) -> list[ResourceOutcome]:
    """Attempt to deregister every ``DELETED``/``NOT_FOUND`` outcome. Any
    deregister failure flips the outcome to ``FAILED("firestore")`` so the
    run's status reflects the incomplete cleanup and stays at ``running``
    for next-cycle retry.
    """
    result: list[ResourceOutcome] = []
    for record in outcomes:
        if record.outcome.status not in (DeleteStatus.DELETED, DeleteStatus.NOT_FOUND):
            result.append(record)
            continue
        if _safe_deregister(record.resource_id):
            result.append(record)
            continue
        result.append(
            ResourceOutcome(
                record.resource_id,
                record.resource,
                DeleteOutcome(
                    DeleteStatus.FAILED,
                    error="deregister failed",
                    error_class="firestore",
                ),
            )
        )
    return result


def _process_k8s(
    resources: dict[str, Resource],
    location: dict[str, ClusterInfo],
    run_clusters: list[ClusterInfo],
) -> list[ResourceOutcome]:
    outcomes: list[ResourceOutcome] = []
    for resource_id, resource in resources.items():
        if resource.type not in K8S_DELETERS:
            continue
        clusters_to_try = (
            [location[resource_id]] if resource_id in location else list(run_clusters)
        )
        outcomes.append(
            ResourceOutcome(
                resource_id,
                resource,
                _delete_across_clusters(resource, clusters_to_try),
            )
        )
    return outcomes


def _delete_across_clusters(resource: Resource, clusters: list[ClusterInfo]) -> DeleteOutcome:
    """Attempt deletion on each cluster; return DELETED if any succeeds,
    else the last FAILED outcome, else NOT_FOUND.
    """
    last_failed: DeleteOutcome | None = None
    for cluster in clusters:
        outcome = _try_delete_in_cluster(resource, cluster)
        if outcome.status == DeleteStatus.DELETED:
            return outcome
        if outcome.status == DeleteStatus.FAILED:
            last_failed = outcome
    if last_failed is not None:
        return last_failed
    return DeleteOutcome(DeleteStatus.NOT_FOUND)


def _try_delete_in_cluster(resource: Resource, cluster: ClusterInfo) -> DeleteOutcome:
    """Build a cluster context and dispatch the resource's deleter."""
    try:
        k8s_ctx = build_k8s_context(cluster)
    except GoogleAPICallError as exc:
        return DeleteOutcome(DeleteStatus.FAILED, error=str(exc), error_class="cluster_auth")
    return K8S_DELETERS[resource.type](resource, k8s_ctx)


def _process_sqs(resources: dict[str, Resource]) -> list[ResourceOutcome]:
    sqs_by_region: dict[str, list[tuple[str, Resource]]] = defaultdict(list)
    for resource_id, resource in resources.items():
        if resource.type in SQS_DELETERS:
            sqs_by_region[resource.region or ""].append((resource_id, resource))
    outcomes: list[ResourceOutcome] = []
    for region, region_resources in sqs_by_region.items():
        sqs_ctx = build_sqs_context(region)
        for resource_id, resource in region_resources:
            outcome = SQS_DELETERS[resource.type](resource, sqs_ctx)
            outcomes.append(ResourceOutcome(resource_id, resource, outcome))
    return outcomes


def _unknown_type_outcomes(
    resources: dict[str, Resource],
    handled: list[ResourceOutcome],
) -> list[ResourceOutcome]:
    """Surface any resource whose type wasn't routed to a k8s or SQS deleter.

    Silently skipping unrecognized types lets an orphan ``RESOURCE_DB`` row
    survive every cycle: ``fully_succeeded`` becomes ``True`` vacuously,
    ``STATE=TIMEDOUT`` gets written, and the row keeps the run in
    ``resources_by_run`` forever. Convert to FAILED so the run stays at
    ``state=running`` for inspection.
    """
    handled_ids = {o.resource_id for o in handled}
    return [
        ResourceOutcome(
            rid,
            r,
            DeleteOutcome(
                DeleteStatus.FAILED,
                error=f"no deleter registered for resource type {r.type!r}",
                error_class="unknown_type",
            ),
        )
        for rid, r in resources.items()
        if rid not in handled_ids
    ]


def cleanup_run(
    run_id: str,
    resources_raw: dict[str, dict[str, Any]],
    zetta_user: str,
    timestamp: float,
    heartbeat: float,
    current_state: str,
    clusters: list[ClusterInfo],
    gc_state: RunGCState,
) -> CleanupReport:
    """Tear down all registered resources for a stale run.

    :param run_id: Run id used as key into ``resources_raw``.
    :param resources_raw: Raw ``RESOURCE_DB`` rows for this run keyed by
        resource id.
    :param zetta_user: Run owner.
    :param timestamp: Unix epoch the run was registered.
    :param heartbeat: Unix epoch of the run's last heartbeat.
    :param current_state: The run's current ``RUN_DB`` state. On full
        success, only a ``"running"`` state is transitioned to
        ``"timedout"``; terminal states (``completed`` / ``failed`` /
        ``timedout``) are preserved.
    :param clusters: Clusters the run registered.
    :param gc_state: Previously persisted GC state for this run (or a
        default :class:`RunGCState` if first time seeing it).
    """
    resources = {rid: Resource(**raw) for rid, raw in resources_raw.items()}
    k8s_resources = {rid: r for rid, r in resources.items() if r.type in K8S_DELETERS}

    location, cluster_failures = discover_locations(clusters, k8s_resources)
    outcomes: list[ResourceOutcome] = []
    outcomes.extend(_process_k8s(resources, location, clusters))
    outcomes.extend(_process_sqs(resources))
    outcomes.extend(_unknown_type_outcomes(resources, outcomes))
    outcomes = _apply_deregisters(outcomes)

    error_class = _dominant_error_class(outcomes, cluster_failures)
    report = CleanupReport(
        run_id=run_id,
        zetta_user=zetta_user,
        timestamp=timestamp,
        heartbeat=heartbeat,
        clusters=list(clusters),
        outcomes=outcomes,
        cluster_failures=cluster_failures,
        error_class=error_class,
    )

    _log_run_summary(report)

    if report.fully_succeeded:
        if current_state == RunState.RUNNING.value:
            update_run_info(run_id, {RunInfo.STATE.value: RunState.TIMEDOUT.value})
        else:
            logger.info(
                f"run {run_id}: cleanup complete; preserving terminal state {current_state!r}"
            )
        purge_run_state(run_id)
    else:
        now = time.time()
        save_state(
            run_id,
            RunGCState(
                last_error_class=error_class,
                last_notify_error_class=gc_state.last_notify_error_class,
                failure_cycles=gc_state.failure_cycles + 1,
                last_failure_ts=now,
                last_attempt_ts=now,
            ),
        )

    return report


def _log_run_summary(report: CleanupReport) -> None:
    deleted: list[str] = []
    not_found: list[str] = []
    failed: list[str] = []
    for ro in report.outcomes:
        label = f"{ro.resource.type}/{ro.resource.name}"
        if ro.outcome.status == DeleteStatus.DELETED:
            deleted.append(label)
        elif ro.outcome.status == DeleteStatus.NOT_FOUND:
            not_found.append(label)
        else:
            failed.append(f"{label} ({ro.outcome.error_class}: {ro.outcome.error})")

    c = report.counts
    lines = [
        f"run {report.run_id} ({report.zetta_user}) [{report.status_label}]: "
        f"{c['total']} total, deleted {c['deleted']}, "
        f"not_found {c['not_found']}, failed {c['failed']}"
    ]
    if report.heartbeat > 0:
        lines.append(f"  stale:     {format_duration(time.time() - report.heartbeat)}")
    if report.clusters:
        lines.append(f"  clusters:  {', '.join(format_cluster(c) for c in report.clusters)}")
    if deleted:
        lines.append(f"  deleted:   {', '.join(deleted)}")
    if not_found:
        lines.append(f"  not_found: {', '.join(not_found)}")
    if failed:
        lines.append(f"  failed:    {', '.join(failed)}")
    for cluster, failure in report.cluster_failures.items():
        lines.append(
            f"  cluster {format_cluster(cluster)}: {failure.error_class}: {failure.error}"
        )
    logger.info("\n".join(lines))


def _stale_run_data() -> tuple[
    dict[str, dict[str, Any]],
    list[str],
    dict[str, dict[str, Any]],
]:
    """Read RESOURCE_DB + RUN_DB once and return ``(resources_by_run,
    stale_run_ids, run_info_by_id)``. ``run_info_by_id`` is keyed by the
    stripped run id so it lines up with the values in ``stale_run_ids``.
    """
    resources_by_run: dict[str, dict[str, Any]] = defaultdict(dict)
    for resource_id, raw in RESOURCE_DB.query().items():
        resources_by_run[str(raw["run_id"])][resource_id] = raw

    candidate_ids = set(resources_by_run.keys())
    candidate_ids.update(RUN_DB.query(column_filter={"state": ["running"]}).keys())
    candidate_list = list(candidate_ids)

    if not candidate_list:
        return resources_by_run, [], {}

    columns = (
        RunInfo.HEARTBEAT.value,
        RunInfo.TIMESTAMP.value,
        RunInfo.ZETTA_USER.value,
        RunInfo.CLUSTERS.value,
        RunInfo.STATE.value,
    )
    try:
        infos = RUN_DB[(candidate_list, columns)]
    except KeyError:
        # FirestoreBackend.read prechecks the single-row case and raises
        # KeyError when the only requested row doesn't exist (e.g. orphan
        # RESOURCE_DB rows for a run that never wrote a RUN_DB row).
        # Normalize: treat the missing row as empty info.
        infos = [{} for _ in candidate_list]

    stale_ids: list[str] = []
    run_info_by_id: dict[str, dict[str, Any]] = {}
    hb_threshold = time.time() - int(os.environ["EXECUTION_HEARTBEAT_LOOKBACK"])
    terminal_states = {
        RunState.COMPLETED.value,
        RunState.FAILED.value,
        RunState.TIMEDOUT.value,
    }
    for candidate_id, info in zip(candidate_list, infos):
        ts = float(info.get(RunInfo.TIMESTAMP.value, 0))
        hb = float(info.get(RunInfo.HEARTBEAT.value, ts))
        state = str(info.get(RunInfo.STATE.value, ""))
        has_resources = bool(resources_by_run.get(candidate_id))

        if state in terminal_states:
            # Terminal-state runs are processed iff they still have
            # RESOURCE_DB rows. Heartbeat is meaningless here: the
            # repeat-timer thread may have written one moments before
            # cancel(), so a fresh hb does not mean the cleanup
            # actually finished. Retry every cycle until RESOURCE_DB
            # for this run is empty.
            if not has_resources:
                continue
        else:
            # Running (or unknown / missing state) — gate on heartbeat
            # staleness so live runs are never touched.
            if hb >= hb_threshold:
                continue

        stale_ids.append(candidate_id)
        run_info_by_id[candidate_id] = dict(info)

    return resources_by_run, stale_ids, run_info_by_id


def _synthetic_failure_report(
    run_id: str,
    info: dict[str, Any],
    exc: BaseException,
    trace: str,
) -> CleanupReport:
    """Build a ``FAILED("gc_internal")`` report when ``cleanup_run`` itself
    raised an unexpected exception. Keeps the cycle's reporting intact —
    other stale runs still get processed and the Slack summary fires —
    while surfacing the GC's own crash for this specific run. The full
    traceback rides along in ``error`` so it shows up in the Slack
    thread reply.
    """
    synthetic = ResourceOutcome(
        resource_id="(gc-internal)",
        resource=Resource(run_id=run_id, type="(gc-internal)", name="cleanup"),
        outcome=DeleteOutcome(
            DeleteStatus.FAILED,
            error=f"{exc!r}\n{trace}",
            error_class="gc_internal",
        ),
    )
    return CleanupReport(
        run_id=run_id,
        zetta_user=str(info.get(RunInfo.ZETTA_USER.value, "")),
        timestamp=float(info.get(RunInfo.TIMESTAMP.value, 0.0)),
        heartbeat=float(info.get(RunInfo.HEARTBEAT.value, 0.0)),
        clusters=_parse_clusters(info.get(RunInfo.CLUSTERS.value)),
        outcomes=[synthetic],
        cluster_failures={},
        error_class="gc_internal",
    )


def main() -> None:  # pragma: no cover
    try:
        _run_cycle()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        trace = traceback.format_exc()
        logger.exception("GC cycle aborted unexpectedly")
        # Best-effort Slack alert so the operator notices even if logs aren't watched.
        try:
            post_message(
                f"GC cycle crashed: {exc!r}\n```\n{trace}\n```",
            )
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("Failed to alert Slack about GC cycle crash")


def _run_cycle() -> None:  # pragma: no cover
    resources_by_run, stale_ids, run_info_by_id = _stale_run_data()
    if not stale_ids:
        post_idle()
        return

    try:
        states_pre = load_states(stale_ids)
    except Exception:  # pylint: disable=broad-exception-caught
        logger.exception(
            "Failed to bulk-load gc-run-state; falling back to defaults so the cycle "
            "still processes and reports per-run outcomes."
        )
        states_pre = {}
    user_resolver = UserResolver()
    reports: list[CleanupReport] = []
    for run_id in stale_ids:
        info = run_info_by_id[run_id]
        try:
            report = cleanup_run(
                run_id=run_id,
                resources_raw=resources_by_run.get(run_id, {}),
                zetta_user=str(info.get(RunInfo.ZETTA_USER.value, "")),
                timestamp=float(info.get(RunInfo.TIMESTAMP.value, 0.0)),
                heartbeat=float(info.get(RunInfo.HEARTBEAT.value, 0.0)),
                current_state=str(info.get(RunInfo.STATE.value, "")),
                clusters=_parse_clusters(info.get(RunInfo.CLUSTERS.value)),
                gc_state=states_pre.get(run_id, RunGCState()),
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            trace = traceback.format_exc()
            logger.exception(f"run {run_id}: unexpected error during cleanup")
            report = _synthetic_failure_report(run_id, info, exc, trace)
        reports.append(report)
    try:
        post_cycle(reports, states_pre, user_resolver)
    except Exception:  # pylint: disable=broad-exception-caught
        logger.exception("Slack post_cycle failed; per-run summaries already logged above.")
