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
from kubernetes.client.exceptions import ApiException

from zetta_utils import log
from zetta_utils.message_queues.sqs import utils as sqs_utils

from .common import ClusterInfo, get_cluster_data
from .deployment import deployment_ctx_mngr

logger = log.get_logger("zetta_utils")

MAX_REPLICAS_ANNOTATION = "zetta.ai/autoscaler-max-replicas"
MIN_REPLICAS_ANNOTATION = "zetta.ai/autoscaler-min-replicas"
SCALE_DOWN_STABILIZATION_SEC_ANNOTATION = "zetta.ai/autoscaler-scale-down-stabilization-sec"


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
        try:
            dep = self.apps_api.read_namespaced_deployment(
                name=self.target.deployment_name, namespace=self.target.namespace
            )
        except ApiException as exc:
            logger.warning(
                f"autoscaler: read_deployment {self.target.deployment_name}: "
                f"{exc.status} {exc.reason}"
            )
            return

        config = self._read_scaling_config(dep)
        if config is None:
            return
        max_replicas, min_replicas, stabilization_sec = config

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
        try:
            self.apps_api.patch_namespaced_deployment(
                name=self.target.deployment_name,
                namespace=self.target.namespace,
                body={"spec": {"replicas": replicas}},
            )
            logger.info(
                f"autoscaler: {self.target.deployment_name}: " f"{current} -> {replicas} replicas"
            )
        except ApiException as exc:
            logger.warning(
                f"autoscaler: patch_deployment {self.target.deployment_name}: "
                f"{exc.status} {exc.reason}"
            )


def _run_loop(
    target: AutoscaleTarget,
    cluster_info: ClusterInfo,
    stop_event: threading.Event,
    poll_interval_sec: int,
) -> None:
    """Daemon body: tick the scaler until ``stop_event`` is set."""
    configuration, _ = get_cluster_data(cluster_info)
    k8s_client.Configuration.set_default(configuration)
    apps_api = k8s_client.AppsV1Api()
    scaler = _GroupScaler(target=target, apps_api=apps_api)
    while not stop_event.wait(poll_interval_sec):
        try:
            scaler.tick()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning(f"autoscaler: tick failed for {target.deployment_name}: {exc}")


@contextmanager
def autoscaling_deployment_ctx_mngr(
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
    thread = threading.Thread(
        target=_run_loop,
        args=(target, cluster_info, stop_event, poll_interval_sec),
        daemon=True,
    )
    with deployment_ctx_mngr(run_id, cluster_info, deployment, secrets, namespace=namespace):
        thread.start()
        try:
            yield
        finally:
            stop_event.set()
            thread.join(timeout=poll_interval_sec * 2)
