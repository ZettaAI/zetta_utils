"""
Helpers for k8s deployments.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, List, Literal, Optional

from kubernetes import client as k8s_client
from zetta_utils import builder, log
from zetta_utils.common import RepeatTimer
from zetta_utils.mazepa import SemaphoreType
from zetta_utils.run.resource import (
    Resource,
    ResourceTypes,
    deregister_resource,
    register_resource,
)

from .common import ClusterInfo, get_cluster_data, get_mazepa_worker_command
from .pod import get_mazepa_pod_spec, stream_pod_logs
from .secret import secrets_ctx_mngr

logger = log.get_logger("zetta_utils")


def _get_mazepa_deployment(
    name: str,
    image: str,
    command: str,
    replicas: int,
    resources: Dict[str, int | float | str],
    labels: Dict[str, str],
    env_secret_mapping: Dict[str, str],
    resource_requests: Optional[Dict[str, int | float | str]] = None,
    provisioning_model: Literal["standard", "spot"] = "spot",
    gpu_accelerator_type: str | None = None,
    adc_available: bool = False,
    cave_secret_available: bool = False,
    required_zones: list[str] | None = None,
    preferred_zones: list[str] | None = None,
) -> k8s_client.V1Deployment:
    name = f"run-{name}"
    pod_spec = get_mazepa_pod_spec(
        image=image,
        command=command,
        resources=resources,
        env_secret_mapping=env_secret_mapping,
        provisioning_model=provisioning_model,
        resource_requests=resource_requests,
        gpu_accelerator_type=gpu_accelerator_type,
        adc_available=adc_available,
        cave_secret_available=cave_secret_available,
        required_zones=required_zones,
        preferred_zones=preferred_zones,
    )

    pod_template = k8s_client.V1PodTemplateSpec(
        metadata=k8s_client.V1ObjectMeta(labels=labels),
        spec=pod_spec,
    )

    deployment_spec = k8s_client.V1DeploymentSpec(
        progress_deadline_seconds=600,
        replicas=replicas,
        selector=k8s_client.V1LabelSelector(match_labels=labels),
        strategy=k8s_client.V1DeploymentStrategy(
            type="RollingUpdate",
            rolling_update=k8s_client.V1RollingUpdateDeployment(
                max_surge="25%", max_unavailable="25%"
            ),
        ),
        template=pod_template,
    )

    deployment = k8s_client.V1Deployment(
        metadata=k8s_client.V1ObjectMeta(name=name, labels=labels),
        spec=deployment_spec,
    )
    return deployment


def get_mazepa_worker_deployment(  # pylint: disable=too-many-locals
    run_id: str,
    image: str,
    task_queue_spec: dict[str, Any],
    outcome_queue_spec: dict[str, Any],
    replicas: int,
    resources: Dict[str, int | float | str],
    env_secret_mapping: Dict[str, str],
    labels: Optional[Dict[str, str]] = None,
    resource_requests: Optional[Dict[str, int | float | str]] = None,
    num_procs: int = 1,
    semaphores_spec: dict[SemaphoreType, int] | None = None,
    provisioning_model: Literal["standard", "spot"] = "spot",
    gpu_accelerator_type: str | None = None,
    adc_available: bool = False,
    cave_secret_available: bool = False,
    required_zones: list[str] | None = None,
    preferred_zones: list[str] | None = None,
):
    if labels is None:
        labels_final = {"run_id": run_id}
    else:
        labels_final = labels

    worker_command = get_mazepa_worker_command(
        task_queue_spec, outcome_queue_spec, num_procs, semaphores_spec
    )
    logger.debug(f"Making a deployment with worker command: '{worker_command}'")

    return _get_mazepa_deployment(
        name=run_id,
        image=image,
        replicas=replicas,
        command=worker_command,
        resources=resources,
        labels=labels_final,
        env_secret_mapping=env_secret_mapping,
        resource_requests=resource_requests,
        provisioning_model=provisioning_model,
        gpu_accelerator_type=gpu_accelerator_type,
        adc_available=adc_available,
        cave_secret_available=cave_secret_available,
        required_zones=required_zones,
        preferred_zones=preferred_zones,
    )


def get_deployment(
    name: str,
    pod_spec: k8s_client.V1PodSpec,
    replicas: int,
    labels: Optional[Dict[str, str]] = None,
    revision_history_limit: Optional[int] = 10,
) -> k8s_client.V1Deployment:
    name = f"run-{name}"
    labels = labels or {"app": name}
    pod_template = k8s_client.V1PodTemplateSpec(
        metadata=k8s_client.V1ObjectMeta(labels=labels),
        spec=pod_spec,
    )

    deployment_spec = k8s_client.V1DeploymentSpec(
        replicas=replicas,
        revision_history_limit=revision_history_limit,
        selector=k8s_client.V1LabelSelector(match_labels=labels),
        template=pod_template,
    )

    deployment = k8s_client.V1Deployment(
        metadata=k8s_client.V1ObjectMeta(name=name, labels=labels),
        spec=deployment_spec,
    )

    return deployment


@builder.register("k8s_deployment_ctx_mngr")
@contextmanager
def deployment_ctx_mngr(
    run_id: str,
    cluster_info: ClusterInfo,
    deployment: k8s_client.V1Deployment,
    secrets: List[k8s_client.V1Secret],
    namespace: str = "default",
    stream_logs: bool = False,
    tail_lines: int | None = None,
):
    def _stream_deployment_logs():
        dep_selector = ",".join(
            f"{k}={v}" for k, v in deployment.spec.selector.match_labels.items()
        )
        _name = deployment.metadata.name
        stream_pod_logs(
            cluster_info,
            dep_selector=dep_selector,
            namespace=namespace,
            prefix=f"{_name[:7]}...{_name[-7:]}",
            tail_lines=tail_lines,
        )

    configuration, _ = get_cluster_data(cluster_info)
    k8s_client.Configuration.set_default(configuration)
    k8s_apps_v1_api = k8s_client.AppsV1Api()
    log_streamer = None

    with secrets_ctx_mngr(run_id, secrets, cluster_info, namespace=namespace):
        logger.info(f"Creating k8s deployment `{deployment.metadata.name}`")
        k8s_apps_v1_api.create_namespaced_deployment(body=deployment, namespace=namespace)
        _id = register_resource(
            Resource(
                run_id,
                ResourceTypes.K8S_DEPLOYMENT.value,
                deployment.metadata.name,
            )
        )

        if stream_logs:
            log_streamer = RepeatTimer(15, _stream_deployment_logs)
            log_streamer.start()

        try:
            yield
        finally:
            # new configuration to refresh expired tokens (long running executions)
            configuration, _ = get_cluster_data(cluster_info)
            k8s_client.Configuration.set_default(configuration)

            # need to create a new client for the above to take effect
            k8s_apps_v1_api = k8s_client.AppsV1Api()
            logger.info(f"Deleting k8s deployment `{deployment.metadata.name}`")
            if log_streamer:
                log_streamer.cancel()
            k8s_apps_v1_api.delete_namespaced_deployment(
                name=deployment.metadata.name, namespace=namespace
            )
            deregister_resource(_id)
