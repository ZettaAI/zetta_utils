"""
Helpers for k8s pod.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from kubernetes import client as k8s_client
from zetta_utils import log
from zetta_utils.cloud_management.resource_allocation.k8s.volume import (
    get_common_volume_mounts,
    get_common_volumes,
)

from .secret import get_worker_env_vars

logger = log.get_logger("zetta_utils")


def get_pod_spec(
    name: str,
    image: str,
    command: List[str],
    command_args: List[str],
    resources: Optional[Dict[str, int | float | str]] = None,
    dns_policy: Optional[str] = "Default",
    envs: Optional[List[k8s_client.V1EnvVar]] = None,
    env_secret_mapping: Optional[Dict[str, str]] = None,
    hostname: Optional[str] = None,
    host_network: Optional[bool] = False,
    host_aliases: Optional[List[k8s_client.V1HostAlias]] = None,
    image_pull_policy: Optional[str] = "IfNotPresent",
    node_selector: Optional[Dict[str, str]] = None,
    restart_policy: Optional[str] = "Always",
    tolerations: Optional[List[k8s_client.V1Toleration]] = None,
    volumes: Optional[List[k8s_client.V1Volume]] = None,
    volume_mounts: Optional[List[k8s_client.V1VolumeMount]] = None,
    resource_requests: Optional[Dict[str, int | float | str]] = None,
) -> k8s_client.V1PodSpec:
    name = f"run-{name}"
    envs = envs or []
    host_aliases = host_aliases or []
    tolerations = tolerations or []
    volumes = volumes or []
    volume_mounts = volume_mounts or []
    ports = [k8s_client.V1ContainerPort(container_port=29400)]
    container = k8s_client.V1Container(
        command=command,
        args=command_args,
        env=envs + get_worker_env_vars(env_secret_mapping),
        name=name,
        image=image,
        image_pull_policy=image_pull_policy,
        ports=ports,
        resources=k8s_client.V1ResourceRequirements(
            limits=resources,
            requests=resource_requests or resources,
        ),
        termination_message_path="/dev/termination-log",
        termination_message_policy="File",
        volume_mounts=volume_mounts,
    )

    return k8s_client.V1PodSpec(
        containers=[container],
        dns_policy=dns_policy,
        hostname=hostname,
        host_network=host_network,
        host_aliases=host_aliases,
        node_selector=node_selector,
        restart_policy=restart_policy,
        scheduler_name="default-scheduler",
        security_context={},
        termination_grace_period_seconds=30,
        tolerations=tolerations,
        volumes=volumes,
    )


def get_mazepa_pod_spec(
    image: str,
    command: str,
    resources: Optional[Dict[str, int | float | str]] = None,
    env_secret_mapping: Optional[Dict[str, str]] = None,
    provisioning_model: Literal["standard", "spot"] = "spot",
    resource_requests: Optional[Dict[str, int | float | str]] = None,
) -> k8s_client.V1PodSpec:
    schedule_toleration = k8s_client.V1Toleration(
        key="worker-pool", operator="Equal", value="true", effect="NoSchedule"
    )

    return get_pod_spec(
        name="zutils-worker",
        image=image,
        command=["/bin/sh"],
        command_args=["-c", command],
        resources=resources,
        env_secret_mapping=env_secret_mapping,
        node_selector={"cloud.google.com/gke-provisioning": provisioning_model},
        restart_policy="Never",
        tolerations=[schedule_toleration],
        volumes=get_common_volumes(),
        volume_mounts=get_common_volume_mounts(),
        resource_requests=resource_requests,
    )
