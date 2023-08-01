"""
Helpers for k8s deployments.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from kubernetes import client as k8s_client  # type: ignore
from zetta_utils import builder, log

from ..resource_tracker import (
    ExecutionResource,
    ExecutionResourceTypes,
    register_execution_resource,
)
from .common import ClusterInfo, get_cluster_data, get_mazepa_worker_command
from .pod import get_pod_spec
from .secret import secrets_ctx_mngr

logger = log.get_logger("zetta_utils")


def get_deployment_spec(
    name: str,
    image: str,
    command: str,
    replicas: int,
    resources: Dict[str, int | float | str],
    labels: Dict[str, str],
    env_secret_mapping: Dict[str, str],
) -> k8s_client.V1Deployment:
    schedule_toleration = k8s_client.V1Toleration(
        key="worker-pool", operator="Equal", value="true", effect="NoSchedule"
    )

    pod_spec = get_pod_spec(
        name="zutils-worker",
        image=image,
        command=["/bin/sh"],
        command_args=["-c", command],
        resources=resources,
        env_secret_mapping=env_secret_mapping,
        tolerations=[schedule_toleration],
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
    execution_id: str,
    image: str,
    task_queue_spec: dict[str, Any],
    outcome_queue_spec: dict[str, Any],
    replicas: int,
    resources: Dict[str, int | float | str],
    env_secret_mapping: Dict[str, str],
    labels: Optional[Dict[str, str]] = None,
):
    if labels is None:
        labels_final = {"execution_id": execution_id}
    else:
        labels_final = labels

    worker_command = get_mazepa_worker_command(task_queue_spec, outcome_queue_spec)
    logger.debug(f"Making a deployment with worker command: '{worker_command}'")

    return get_deployment_spec(
        name=execution_id,
        image=image,
        replicas=replicas,
        command=worker_command,
        resources=resources,
        labels=labels_final,
        env_secret_mapping=env_secret_mapping,
    )


def get_deployment(
    name: str,
    pod_spec: k8s_client.V1PodSpec,
    replicas: int,
    labels: Optional[Dict[str, str]] = None,
    revision_history_limit: Optional[int] = 10,
) -> k8s_client.V1Deployment:
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
    execution_id: str,
    cluster_info: ClusterInfo,
    deployment: k8s_client.V1Deployment,
    secrets: List[k8s_client.V1Secret],
    namespace: Optional[str] = "default",
):
    configuration, _ = get_cluster_data(cluster_info)
    k8s_client.Configuration.set_default(configuration)
    k8s_apps_v1_api = k8s_client.AppsV1Api()

    with secrets_ctx_mngr(execution_id, secrets, cluster_info):
        logger.info(f"Creating k8s deployment `{deployment.metadata.name}`")
        k8s_apps_v1_api.create_namespaced_deployment(body=deployment, namespace=namespace)
        register_execution_resource(
            ExecutionResource(
                execution_id,
                ExecutionResourceTypes.K8S_DEPLOYMENT.value,
                deployment.metadata.name,
            )
        )

        try:
            yield
        finally:
            # new configuration to refresh expired tokens (long running executions)
            configuration, _ = get_cluster_data(cluster_info)
            k8s_client.Configuration.set_default(configuration)

            # need to create a new client for the above to take effect
            k8s_apps_v1_api = k8s_client.AppsV1Api()
            logger.info(f"Deleting k8s deployment `{deployment.metadata.name}`")
            k8s_apps_v1_api.delete_namespaced_deployment(
                name=deployment.metadata.name, namespace=namespace
            )
