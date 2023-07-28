"""
Helpers for k8s pod.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from kubernetes import client as k8s_client  # type: ignore
from zetta_utils import log

from .secret import get_worker_env_vars

logger = log.get_logger("zetta_utils")


def get_pod_spec(
    name: str,
    image: str,
    command: List[str],
    command_args: List[str],
    resources: Dict[str, int | float | str],
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
) -> k8s_client.V1PodSpec:

    envs = envs or []
    host_aliases = host_aliases or []
    tolerations = tolerations or []
    volumes = volumes or []
    volume_mounts = volume_mounts or []

    volume_mounts += [
        k8s_client.V1VolumeMount(mount_path="/dev/shm", name="dshm"),
        k8s_client.V1VolumeMount(mount_path="/tmp", name="tmp"),
    ]

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
            requests=resources,
            limits=resources,
        ),
        termination_message_path="/dev/termination-log",
        termination_message_policy="File",
        volume_mounts=volume_mounts,
    )

    dshm = k8s_client.V1Volume(
        name="dshm", empty_dir=k8s_client.V1EmptyDirVolumeSource(medium="Memory")
    )
    tmp = k8s_client.V1Volume(
        name="tmp", empty_dir=k8s_client.V1EmptyDirVolumeSource(medium="Memory")
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
        volumes=volumes + [dshm, tmp],
    )
