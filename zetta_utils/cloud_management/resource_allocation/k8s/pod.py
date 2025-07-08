"""
Helpers for k8s pod.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Mapping, Optional

from kubernetes import client as k8s_client
from zetta_utils.cloud_management.resource_allocation import k8s

from .secret import get_worker_env_vars


def get_pod_spec(
    name: str,
    image: str,
    command: str,
    resources: Optional[Dict[str, int | float | str]] = None,
    dns_policy: Optional[str] = "Default",
    envs: Optional[List[k8s_client.V1EnvVar]] = None,
    env_secret_mapping: Optional[Dict[str, str]] = None,
    hostname: Optional[str] = None,
    host_network: Optional[bool] = False,
    host_aliases: Optional[List[k8s_client.V1HostAlias]] = None,
    image_pull_policy: Optional[str] = "IfNotPresent",
    node_selector: Optional[Mapping[str, str]] = None,
    restart_policy: Optional[str] = "Always",
    tolerations: Optional[List[k8s_client.V1Toleration]] = None,
    volumes: Optional[List[k8s_client.V1Volume]] = None,
    volume_mounts: Optional[List[k8s_client.V1VolumeMount]] = None,
    resource_requests: Optional[Dict[str, int | float | str]] = None,
) -> k8s_client.V1PodSpec:
    name = f"run-{name}"
    envs = envs or []
    env_secret_mapping = env_secret_mapping or {}

    try:
        run_id_env = k8s_client.V1EnvVar(name="RUN_ID", value=env_secret_mapping.pop("RUN_ID"))
        envs.append(run_id_env)
    except KeyError:
        ...

    envs.append(
        k8s_client.V1EnvVar(
            name="POD_NAME",
            value_from=k8s_client.V1EnvVarSource(
                field_ref=k8s_client.V1ObjectFieldSelector(field_path="metadata.name")
            ),
        )
    )
    envs.extend(get_worker_env_vars(env_secret_mapping))

    host_aliases = host_aliases or []
    tolerations = tolerations or []
    volumes = volumes or []
    volume_mounts = volume_mounts or []

    volumes.append(
        k8s_client.V1Volume(name="shared-volume", empty_dir=k8s_client.V1EmptyDirVolumeSource())
    )

    volume_mounts.append(k8s_client.V1VolumeMount(name="shared-volume", mount_path="/shared"))
    ports = [k8s_client.V1ContainerPort(container_port=29400)]
    container = k8s_client.V1Container(
        command=["/bin/bash", "-c"],
        args=[command],
        env=envs,
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

    module = "zetta_utils.cloud_management.resource_allocation.k8s.log_pod_runtime"
    cmd = f"python -m {module}"
    sidecar_container = k8s_client.V1Container(
        command=["/bin/bash", "-c"],
        args=[f"{cmd}; while [ ! -f /shared/done.txt ]; do echo sleeping && sleep 1; done; {cmd}"],
        env=envs,
        name="log-pod-runtime",
        image=image,
        termination_message_path="/dev/termination-log",
        termination_message_policy="File",
        volume_mounts=volume_mounts,
    )

    return k8s_client.V1PodSpec(
        containers=[container, sidecar_container],
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
    restart_policy: Literal["Always", "Never"] = "Always",
    gpu_accelerator_type: str | None = None,
    adc_available: bool = False,
    cave_secret_available: bool = False,
) -> k8s_client.V1PodSpec:
    schedule_toleration = k8s_client.V1Toleration(
        key="worker-pool", operator="Equal", value="true", effect="NoSchedule"
    )

    node_selector: dict[str, str] = {"cloud.google.com/gke-provisioning": provisioning_model}
    if gpu_accelerator_type:
        node_selector["cloud.google.com/gke-accelerator"] = gpu_accelerator_type

    envs = []
    if adc_available:
        envs.append(
            k8s_client.V1EnvVar(
                name="GOOGLE_APPLICATION_CREDENTIALS", value=k8s.volume.ADC_MOUNT_PATH
            )
        )

    return get_pod_spec(
        name="zutils-worker",
        image=image,
        command=command,
        resources=resources,
        envs=envs,
        env_secret_mapping=env_secret_mapping,
        node_selector=node_selector,
        restart_policy=restart_policy,
        tolerations=[schedule_toleration],
        volumes=k8s.volume.get_common_volumes(cave_secret_available=cave_secret_available),
        volume_mounts=k8s.volume.get_common_volume_mounts(
            cave_secret_available=cave_secret_available
        ),
        resource_requests=resource_requests,
    )
