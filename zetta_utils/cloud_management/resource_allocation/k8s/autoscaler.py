"""Master-side autoscaler that scales ``Deployment.spec.replicas`` to track SQS depth.

Public surface is :func:`autoscaling_deployment_ctx_mngr`, a context manager
that creates a Deployment and runs a daemon thread for its lifetime that
PATCHes the Deployment's replica count to match the queue depth.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager

import attrs
import kubernetes.client as k8s_client
from google.api_core.exceptions import FailedPrecondition, GoogleAPICallError

from zetta_utils import log
from zetta_utils.message_queues.sqs import utils as sqs_utils

from . import gke
from .common import ClusterInfo, get_cluster_data
from .deployment import deployment_ctx_mngr
from .pod import watch_for_triggered_scale_up

logger = log.get_logger("zetta_utils")

MAX_REPLICAS_ANNOTATION = "zetta.ai/autoscaler-max-replicas"
MIN_REPLICAS_ANNOTATION = "zetta.ai/autoscaler-min-replicas"
SCALE_DOWN_STABILIZATION_SEC_ANNOTATION = "zetta.ai/autoscaler-scale-down-stabilization-sec"

# Cross-group node-pool nudger state.
_pool_state: dict[str, _PoolState] = {}
_mig_to_pool: dict[str, str] = {}
_mig_to_pool_refreshed_at: dict[tuple[str, str, str], float] = {}
_MIG_REFRESH_MIN_GAP_SEC = 60

# Process-wide ``AppsV1Api`` shared across the per-group autoscaler and
# nudger threads. Built from ``get_cluster_data(cluster_info)`` (programmatic
# auth via ADC) so it works regardless of whether the caller has a local
# kubeconfig. Reset on any tick failure so the next iteration rebuilds
# against fresh credentials.
_apps_v1_api: k8s_client.AppsV1Api | None = None


def _get_apps_v1_api(cluster_info: ClusterInfo) -> k8s_client.AppsV1Api:
    global _apps_v1_api  # pylint: disable=global-statement
    if _apps_v1_api is None:
        configuration, _ = get_cluster_data(cluster_info)
        k8s_client.Configuration.set_default(configuration)
        _apps_v1_api = k8s_client.AppsV1Api()
    return _apps_v1_api


def _reset_apps_v1_api() -> None:
    global _apps_v1_api  # pylint: disable=global-statement
    _apps_v1_api = None


@attrs.mutable
class _PoolState:
    """Per-pool nudger state shared across all groups."""

    last_nudge_at: float | None = None


@attrs.mutable
class _GroupNudgeState:
    """Per-group state shared between the TriggeredScaleUp watcher and nudger.

    ``attempted_pool`` is the pool the cluster autoscaler last said it would
    scale; ``attempted_target_per_zone`` is the per-zone node count CA picked.
    The nudger applies that target directly via SetNodePoolSize so we
    bypass CA backoff without re-doing CA's resource math.
    """

    attempted_pool: str | None = None
    attempted_target_per_zone: int | None = None


def _require_gke_cluster_info(cluster_info: ClusterInfo) -> tuple[str, str, str]:
    """Return ``(project, region, name)`` from the cluster info, raising if unset.

    The nudger is GKE-specific; ``ClusterInfo.project`` and ``.region`` may
    legitimately be ``None`` for non-GKE backends, but in that case the
    nudger should not have been started in the first place.
    """
    project = cluster_info.project
    region = cluster_info.region
    if project is None or region is None:
        raise RuntimeError("node-pool nudger requires cluster_info with project and region set")
    return project, region, cluster_info.name


def _refresh_mig_to_pool(cluster_info: ClusterInfo) -> None:
    """Re-list node pools and rebuild the MIG → pool name mapping in place."""
    project, region, cluster = _require_gke_cluster_info(cluster_info)
    pools = gke.list_node_pools(project, region, cluster)
    for pool in pools:
        for url in pool.instance_group_urls:
            mig = url.rsplit("/", 1)[-1]
            _mig_to_pool[mig] = pool.name
    _mig_to_pool_refreshed_at[(project, region, cluster)] = time.monotonic()


def _resolve_pool(mig_name: str, cluster_info: ClusterInfo) -> str | None:
    """Look up the pool that owns ``mig_name``.

    Accepts either a short MIG name or a full instanceGroups URL; the cache is
    keyed by the short name. On cache miss (e.g. NAP just created a new pool),
    refresh the MIG → pool mapping at most once per
    :data:`_MIG_REFRESH_MIN_GAP_SEC`. Returns ``None`` if the MIG is still
    unknown after the refresh — the caller drops the event.
    """
    mig_name = mig_name.rsplit("/", 1)[-1]
    pool = _mig_to_pool.get(mig_name)
    if pool is not None:
        return pool
    key = _require_gke_cluster_info(cluster_info)
    last = _mig_to_pool_refreshed_at.get(key, 0.0)
    if time.monotonic() - last < _MIG_REFRESH_MIN_GAP_SEC:
        return None
    _refresh_mig_to_pool(cluster_info)
    return _mig_to_pool.get(mig_name)


def _nudge_pool(
    pool_name: str,
    target_per_zone: int,
    cluster_info: ClusterInfo,
    nudge_min_gap_sec: int,
) -> None:
    """Resize ``pool_name`` to ``target_per_zone`` per-zone nodes.

    The target comes from the cluster autoscaler's own ``TriggeredScaleUp``
    decision, so we defer resource math to it and only intervene to bypass
    its post-RESOURCE_POOL_EXHAUSTED backoff. Honors per-pool cool-off and
    never shrinks (CA may have already grown the pool past its event-time
    target). Fire-and-forget: never blocks on the long-running operation.
    """
    state = _pool_state.setdefault(pool_name, _PoolState())
    now = time.monotonic()
    if state.last_nudge_at is not None and now - state.last_nudge_at < nudge_min_gap_sec:
        return

    project, region, cluster = _require_gke_cluster_info(cluster_info)
    pools = gke.list_node_pools(project, region, cluster)
    pool = next((p for p in pools if p.name == pool_name), None)
    if pool is None:
        logger.warning(f"node-pool nudge: pool {pool_name} not found, skipping")
        return

    current = pool.initial_node_count
    num_zones = max(1, len(pool.locations))
    target = max(current, target_per_zone)
    if target <= current:
        return

    added = (target - current) * num_zones
    new_total = target * num_zones
    try:
        gke.resize_node_pool(project, region, cluster, pool_name, target)
        logger.info(
            f"node-pool nudge: {pool_name}: adding {added} nodes "
            f"(now {new_total} total) per CA TriggeredScaleUp target"
        )
        state.last_nudge_at = now
    except FailedPrecondition:
        logger.info(f"node-pool nudge: {pool_name} concurrent op, retry next cycle")
    except GoogleAPICallError as exc:
        logger.warning(f"node-pool nudge: {pool_name} failed: {exc}")


@attrs.frozen
class AutoscaleTarget:
    """Identity of one autoscaling target.

    Replica bounds and the scale-down stabilization window are stored on the
    Deployment as annotations (see :data:`MAX_REPLICAS_ANNOTATION`,
    :data:`MIN_REPLICAS_ANNOTATION`,
    :data:`SCALE_DOWN_STABILIZATION_SEC_ANNOTATION`). Reading them on every
    tick lets ``kubectl annotate`` (or the run-update CLI) adjust the
    autoscaler's behavior at runtime without restarting the master.

    :param deployment_name: name of the k8s Deployment to scale.
    :param queue_name: SQS queue whose depth drives scaling.
    :param region_name: AWS region of ``queue_name`` (passed to boto3).
    :param namespace: k8s namespace the Deployment lives in. Defaults to
        ``"default"``.
    """

    deployment_name: str
    queue_name: str
    region_name: str
    namespace: str = "default"


@attrs.mutable
class _GroupScaler:
    """Per-target tick state. Tracks the scale-down stabilization timer."""

    target: AutoscaleTarget
    apps_api: k8s_client.AppsV1Api
    _scale_down_pending_since: float | None = None

    def tick(self) -> None:
        dep = self.apps_api.read_namespaced_deployment(
            name=self.target.deployment_name, namespace=self.target.namespace
        )

        scaling_config = self._read_scaling_config(dep)
        if scaling_config is None:
            return
        max_replicas, min_replicas, stabilization_sec = scaling_config

        visible, in_flight = sqs_utils.get_queue_depth(
            self.target.queue_name, self.target.region_name
        )
        desired = max(min_replicas, min(max_replicas, visible + in_flight))
        current = dep.spec.replicas or 0

        if desired > current:
            self._patch(desired, current)
            self._scale_down_pending_since = None
        elif desired == current:
            self._scale_down_pending_since = None
        else:
            now = time.monotonic()
            if self._scale_down_pending_since is None:
                self._scale_down_pending_since = now
            elif now - self._scale_down_pending_since >= stabilization_sec:
                self._patch(desired, current)
                self._scale_down_pending_since = None

    def _read_scaling_config(self, dep: k8s_client.V1Deployment) -> tuple[int, int, int] | None:
        """Pull (max, min, stabilization_sec) from Deployment annotations.

        Returns ``None`` (and logs an error) if any required annotation is
        missing or malformed. Caller skips the tick on ``None`` rather than
        scaling against partial config.
        """
        annotations = dep.metadata.annotations or {}
        try:
            return (
                int(annotations[MAX_REPLICAS_ANNOTATION]),
                int(annotations[MIN_REPLICAS_ANNOTATION]),
                int(annotations[SCALE_DOWN_STABILIZATION_SEC_ANNOTATION]),
            )
        except KeyError as exc:
            logger.error(
                f"autoscaler: {self.target.deployment_name} missing annotation "
                f"{exc.args[0]}; skipping tick"
            )
            return None
        except ValueError as exc:
            logger.error(
                f"autoscaler: {self.target.deployment_name} has malformed "
                f"scaling annotation: {exc}; skipping tick"
            )
            return None

    def _patch(self, replicas: int, current: int) -> None:
        if replicas == current:
            return
        self.apps_api.patch_namespaced_deployment(
            name=self.target.deployment_name,
            namespace=self.target.namespace,
            body={"spec": {"replicas": replicas}},
        )
        logger.info(
            f"autoscaler: {self.target.deployment_name}: " f"{current} -> {replicas} replicas"
        )


def _run_loop(
    target: AutoscaleTarget,
    cluster_info: ClusterInfo,
    stop_event: threading.Event,
    poll_interval_sec: int,
) -> None:
    """Daemon body: tick the scaler until ``stop_event`` is set.

    On any tick failure the shared ``AppsV1Api`` is reset so the next
    iteration rebuilds it via ``get_cluster_data(cluster_info)`` -
    recovers from token expiry and transient init failures alike.
    """
    scaler: _GroupScaler | None = None
    while not stop_event.wait(poll_interval_sec):
        try:
            if scaler is None:
                scaler = _GroupScaler(target=target, apps_api=_get_apps_v1_api(cluster_info))
            scaler.tick()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning(f"autoscaler: tick failed for {target.deployment_name}: {exc}")
            _reset_apps_v1_api()
            scaler = None


def _run_triggered_scale_up_watcher(
    name_prefix: str,
    group_state: _GroupNudgeState,
    cluster_info: ClusterInfo,
    namespace: str,
    stop_event: threading.Event,
) -> None:
    """Daemon body: subscribe to ``TriggeredScaleUp`` events for one group's pods.

    Each event names the MIG(s) the cluster autoscaler decided to scale up,
    along with CA's chosen per-zone target node count. Resolves the first
    MIG to its pool and stores ``(pool, target_per_zone)`` on
    ``group_state``; the nudge loop applies that target directly.
    """

    def _on_event(pod_name: str, mig_targets: list[tuple[str, int]]) -> None:
        for mig, target in mig_targets:
            pool = _resolve_pool(mig, cluster_info)
            if pool is not None:
                group_state.attempted_pool = pool
                group_state.attempted_target_per_zone = target
                return
        logger.warning(
            f"node-pool nudge: no pool resolved for MIGs "
            f"{[m for m, _ in mig_targets]} (pod={pod_name}); ignoring"
        )

    watch_for_triggered_scale_up(
        name_prefix=name_prefix,
        cluster_info=cluster_info,
        on_event=_on_event,
        namespace=namespace,
        stop_event=stop_event,
    )


def _run_nudge_loop(
    group_state: _GroupNudgeState,
    deployment_name: str,
    namespace: str,
    cluster_info: ClusterInfo,
    stop_event: threading.Event,
    nudge_interval_sec: int,
    nudge_min_gap_sec: int,
) -> None:
    """Daemon body: every ``nudge_interval_sec``, apply CA's last seen scale-up target.

    The watcher records ``(attempted_pool, attempted_target_per_zone)`` from
    each TriggeredScaleUp event. Skip the cycle if the deployment has no
    pending pods (CA already caught up). Otherwise apply the target via
    SetNodePoolSize so we bypass CA's RESOURCE_POOL_EXHAUSTED backoff.
    """
    while not stop_event.wait(nudge_interval_sec):
        try:
            pool = group_state.attempted_pool
            target = group_state.attempted_target_per_zone
            if pool is None or target is None:
                continue
            dep = _get_apps_v1_api(cluster_info).read_namespaced_deployment(
                name=deployment_name, namespace=namespace
            )
            spec_replicas = dep.spec.replicas or 0
            ready_replicas = dep.status.ready_replicas or 0
            if spec_replicas - ready_replicas <= 0:
                group_state.attempted_pool = None
                group_state.attempted_target_per_zone = None
                continue
            try:
                _nudge_pool(pool, target, cluster_info, nudge_min_gap_sec)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning(f"node-pool nudge: {pool} failed: {exc}")
            group_state.attempted_pool = None
            group_state.attempted_target_per_zone = None
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning(f"node-pool nudge: iteration failed for {deployment_name}: {exc}")
            _reset_apps_v1_api()


@contextmanager
def autoscaling_deployment_ctx_mngr(  # pylint: disable=too-many-locals
    run_id: str,
    cluster_info: ClusterInfo,
    deployment: k8s_client.V1Deployment,
    secrets: list[k8s_client.V1Secret],
    queue_name: str,
    region_name: str,
    max_replicas: int,
    min_replicas: int = 0,
    scale_down_stabilization_sec: int = 60,
    poll_interval_sec: int = 15,
    namespace: str = "default",
    nudge_node_pools: bool = True,
    nudge_interval_sec: int = 180,
    nudge_min_gap_sec: int = 60,
):
    """Like :func:`deployment_ctx_mngr` but with an autoscaler bound to the Deployment.

    Lifecycle:

    * On enter: stamps the scaling-knob annotations on the Deployment,
      enters :func:`deployment_ctx_mngr` (creates the Deployment), then
      starts a daemon thread that polls SQS and PATCHes
      ``Deployment.spec.replicas`` to match queue depth.
    * On exit: signals the thread to stop (joining briefly so we don't leak
      a tick that races the Deployment delete), then exits
      :func:`deployment_ctx_mngr` (deletes the Deployment).

    Replica bounds and the stabilization window are written as annotations
    on the Deployment (see :data:`MAX_REPLICAS_ANNOTATION` and friends) and
    re-read each tick, so they can be adjusted at runtime via
    ``kubectl annotate`` or the run-update CLI without restarting the
    master.

    :param run_id: passed through to :func:`deployment_ctx_mngr`.
    :param cluster_info: cluster the Deployment + autoscaler operate on.
    :param deployment: pre-built ``V1Deployment``. Its ``metadata.name`` is
        what the autoscaler patches; its initial ``spec.replicas`` is the
        cold-start size (the autoscaler ramps from there). The scaling
        annotations are stamped onto ``metadata.annotations`` in place
        before the Deployment is created.
    :param secrets: passed through to :func:`deployment_ctx_mngr`.
    :param queue_name: SQS queue whose depth drives scaling.
    :param region_name: AWS region of ``queue_name``.
    :param max_replicas: initial hard upper bound on replicas; written to
        :data:`MAX_REPLICAS_ANNOTATION`.
    :param min_replicas: initial lower bound (``0`` is valid); written to
        :data:`MIN_REPLICAS_ANNOTATION`.
    :param scale_down_stabilization_sec: initial scale-down debounce
        window; written to
        :data:`SCALE_DOWN_STABILIZATION_SEC_ANNOTATION`.
    :param poll_interval_sec: seconds between reconciliation ticks.
    :param namespace: k8s namespace the Deployment lives in.
    """
    if deployment.metadata.annotations is None:
        deployment.metadata.annotations = {}
    deployment.metadata.annotations[MAX_REPLICAS_ANNOTATION] = str(max_replicas)
    deployment.metadata.annotations[MIN_REPLICAS_ANNOTATION] = str(min_replicas)
    deployment.metadata.annotations[SCALE_DOWN_STABILIZATION_SEC_ANNOTATION] = str(
        scale_down_stabilization_sec
    )

    target = AutoscaleTarget(
        deployment_name=deployment.metadata.name,
        queue_name=queue_name,
        region_name=region_name,
        namespace=namespace,
    )
    stop_event = threading.Event()
    threads: list[threading.Thread] = [
        threading.Thread(
            target=_run_loop,
            args=(target, cluster_info, stop_event, poll_interval_sec),
            daemon=True,
        )
    ]

    if nudge_node_pools:
        group_nudge_state = _GroupNudgeState()
        name_prefix = f"{deployment.metadata.name}-"
        threads.append(
            threading.Thread(
                target=_run_triggered_scale_up_watcher,
                args=(
                    name_prefix,
                    group_nudge_state,
                    cluster_info,
                    namespace,
                    stop_event,
                ),
                daemon=True,
            )
        )
        threads.append(
            threading.Thread(
                target=_run_nudge_loop,
                args=(
                    group_nudge_state,
                    deployment.metadata.name,
                    namespace,
                    cluster_info,
                    stop_event,
                    nudge_interval_sec,
                    nudge_min_gap_sec,
                ),
                daemon=True,
            )
        )

    with deployment_ctx_mngr(run_id, cluster_info, deployment, secrets, namespace=namespace):
        for thread in threads:
            thread.start()
        try:
            yield
        finally:
            stop_event.set()
            for thread in reversed(threads):
                thread.join(timeout=poll_interval_sec * 2)
