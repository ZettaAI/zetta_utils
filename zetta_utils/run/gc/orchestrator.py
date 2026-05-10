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
from zetta_utils.run.gc.slack import post_cycle, post_idle
from zetta_utils.run.gc.state import RunGCState, load_states, save_state
from zetta_utils.run.gc.users import UserResolver
from zetta_utils.run.gc.utils import purge_run_state
from zetta_utils.run.resource import RESOURCE_DB, Resource, deregister_resource

logger = get_logger("zetta_utils")

#: Priority order used to pick the dominant ``error_class`` for a run's
#: :class:`CleanupReport`. Cluster-wide failures outrank per-resource ones;
#: auth issues outrank transient 5xx; sqs / k8s_other are last resorts.
_ERROR_CLASS_PRIORITY = (
    "cluster_404",
    "cluster_auth",
    "k8s_auth",
    "k8s_5xx",
    "sqs",
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


def _safe_deregister(resource_id: str) -> None:
    try:
        deregister_resource(resource_id)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning(f"Failed to deregister {resource_id}: {exc}")


def cleanup_run(
    run_id: str,
    resources_raw: dict[str, dict[str, Any]],
    zetta_user: str,
    timestamp: float,
    heartbeat: float,
    clusters: list[ClusterInfo],
    gc_state: RunGCState,
) -> CleanupReport:
    """Tear down all registered resources for a stale run.

    :param run_id: Stripped run id used as key into ``resources_raw``.
    :param resources_raw: Raw ``RESOURCE_DB`` rows for this run keyed by
        resource id.
    :param zetta_user: Run owner.
    :param timestamp: Unix epoch the run was registered.
    :param heartbeat: Unix epoch of the run's last heartbeat.
    :param clusters: Clusters the run registered.
    :param gc_state: Previously persisted GC state for this run (or a
        default :class:`RunGCState` if first time seeing it).
    """
    resources = {rid: Resource(**raw) for rid, raw in resources_raw.items()}
    k8s_resources = {rid: r for rid, r in resources.items() if r.type in K8S_DELETERS}

    location, cluster_failures = discover_locations(clusters, k8s_resources)
    outcomes: list[ResourceOutcome] = []

    for resource_id, resource in resources.items():
        if resource.type not in K8S_DELETERS:
            continue
        if resource_id not in location:
            outcomes.append(
                ResourceOutcome(resource_id, resource, DeleteOutcome(DeleteStatus.NOT_FOUND))
            )
            _safe_deregister(resource_id)
            continue
        cluster = location[resource_id]
        try:
            k8s_ctx = build_k8s_context(cluster)
        except GoogleAPICallError as exc:
            outcomes.append(
                ResourceOutcome(
                    resource_id,
                    resource,
                    DeleteOutcome(DeleteStatus.FAILED, error=str(exc), error_class="cluster_auth"),
                )
            )
            continue
        outcome = K8S_DELETERS[resource.type](resource, k8s_ctx)
        outcomes.append(ResourceOutcome(resource_id, resource, outcome))
        if outcome.status in (DeleteStatus.DELETED, DeleteStatus.NOT_FOUND):
            _safe_deregister(resource_id)

    sqs_by_region: dict[str, list[tuple[str, Resource]]] = defaultdict(list)
    for resource_id, resource in resources.items():
        if resource.type in SQS_DELETERS:
            sqs_by_region[resource.region or ""].append((resource_id, resource))
    for region, region_resources in sqs_by_region.items():
        sqs_ctx = build_sqs_context(region)
        for resource_id, resource in region_resources:
            outcome = SQS_DELETERS[resource.type](resource, sqs_ctx)
            outcomes.append(ResourceOutcome(resource_id, resource, outcome))
            if outcome.status in (DeleteStatus.DELETED, DeleteStatus.NOT_FOUND):
                _safe_deregister(resource_id)

    error_class = _dominant_error_class(outcomes, cluster_failures)
    report = CleanupReport(
        run_id=run_id,
        zetta_user=zetta_user,
        timestamp=timestamp,
        heartbeat=heartbeat,
        outcomes=outcomes,
        cluster_failures=cluster_failures,
        error_class=error_class,
    )

    _log_run_summary(report)

    if report.fully_succeeded:
        update_run_info(run_id, {RunInfo.STATE.value: RunState.TIMEDOUT.value})
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

    lines = [
        f"run {report.run_id} ({report.zetta_user}) [{report.status_label}]: "
        f"deleted {len(deleted)}, not_found {len(not_found)}, failed {len(failed)}"
    ]
    if deleted:
        lines.append(f"  deleted:   {', '.join(deleted)}")
    if not_found:
        lines.append(f"  not_found: {', '.join(not_found)}")
    if failed:
        lines.append(f"  failed:    {', '.join(failed)}")
    for cluster, failure in report.cluster_failures.items():
        lines.append(f"  cluster {cluster.name}: {failure.error_class}: {failure.error}")
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
    )
    infos = RUN_DB[(candidate_list, columns)]

    stale_ids: list[str] = []
    run_info_by_id: dict[str, dict[str, Any]] = {}
    hb_threshold = time.time() - int(os.environ["EXECUTION_HEARTBEAT_LOOKBACK"])
    for candidate_id, info in zip(candidate_list, infos):
        ts = float(info.get(RunInfo.TIMESTAMP.value, 0))
        hb = float(info.get(RunInfo.HEARTBEAT.value, ts))
        if hb >= hb_threshold:
            continue
        stripped = candidate_id[4:] if candidate_id.startswith("run-") else candidate_id
        stale_ids.append(stripped)
        run_info_by_id[stripped] = dict(info)

    return resources_by_run, stale_ids, run_info_by_id


def main() -> None:  # pragma: no cover
    resources_by_run, stale_ids, run_info_by_id = _stale_run_data()
    if not stale_ids:
        post_idle()
        return

    states_pre = load_states(stale_ids)
    user_resolver = UserResolver()
    reports: list[CleanupReport] = []
    for run_id in stale_ids:
        info = run_info_by_id[run_id]
        report = cleanup_run(
            run_id=run_id,
            resources_raw=resources_by_run.get(run_id, {}),
            zetta_user=str(info.get(RunInfo.ZETTA_USER.value, "")),
            timestamp=float(info.get(RunInfo.TIMESTAMP.value, 0.0)),
            heartbeat=float(info.get(RunInfo.HEARTBEAT.value, 0.0)),
            clusters=_parse_clusters(info.get(RunInfo.CLUSTERS.value)),
            gc_state=states_pre.get(run_id, RunGCState()),
        )
        reports.append(report)
    post_cycle(reports, states_pre, user_resolver)
