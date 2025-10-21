"""
Helpers for k8s pod.
"""

from __future__ import annotations

import time
from typing import Dict, List, Literal, Mapping, Optional

from kubernetes.client.exceptions import ApiException

from kubernetes import client as k8s_client
from kubernetes import watch  # type: ignore
from zetta_utils import log
from zetta_utils.cloud_management.resource_allocation import k8s

from .common import ClusterInfo, get_cluster_data
from .secret import get_worker_env_vars

logger = log.get_logger("zetta_utils")


def get_pod_spec(
    name: str,
    image: str,
    command: str,
    resources: Optional[Dict[str, int | float | str]] = None,
    affinity: Optional[k8s_client.V1Affinity] = None,
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
        envs.append(k8s_client.V1EnvVar(name="RUN_ID", value=env_secret_mapping.pop("RUN_ID")))
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

    module = "zetta_utils.cloud_management.resource_allocation.k8s.log_pod_runtime"
    cmd = f"python -m {module}"
    pre_stop_hook = k8s_client.V1Lifecycle(
        pre_stop=k8s_client.V1LifecycleHandler(
            _exec=k8s_client.V1ExecAction(command=["/bin/bash", "-c", cmd])
        )
    )

    ports = [k8s_client.V1ContainerPort(container_port=29400)]
    container = k8s_client.V1Container(
        command=["/bin/bash", "-c"],
        args=[command],
        env=envs,
        name="main",
        image=image,
        image_pull_policy=image_pull_policy,
        lifecycle=pre_stop_hook,
        ports=ports,
        resources=k8s_client.V1ResourceRequirements(
            limits=resources,
            requests=resource_requests or resources,
        ),
        termination_message_path="/dev/termination-log",
        termination_message_policy="File",
        volume_mounts=volume_mounts,
    )

    module = "zetta_utils.cloud_management.resource_allocation.k8s.oom_tracker"
    sidecar_container = k8s_client.V1Container(
        command=["/bin/bash", "-c"],
        args=[f"python -m {module}"],
        env=envs,
        name="runtime",
        image=image,
        termination_message_path="/dev/termination-log",
        termination_message_policy="File",
        volume_mounts=volume_mounts,
    )

    return k8s_client.V1PodSpec(
        affinity=affinity,
        containers=[container, sidecar_container],
        dns_policy=dns_policy,
        hostname=hostname,
        host_network=host_network,
        host_aliases=host_aliases,
        node_selector=node_selector,
        restart_policy=restart_policy,
        scheduler_name="default-scheduler",
        security_context={},
        termination_grace_period_seconds=60,
        tolerations=tolerations,
        volumes=volumes,
    )


def _get_zone_affinities(
    required_zones: list[str] | None = None, preferred_zones: list[str] | None = None
):
    required_zone_affinity = None
    preferred_zone_affinity = None
    if required_zones:
        required_zone_affinity = k8s_client.V1NodeSelector(
            node_selector_terms=[
                k8s_client.V1NodeSelectorTerm(
                    match_expressions=[
                        k8s_client.V1NodeSelectorRequirement(
                            key="topology.kubernetes.io/zone",
                            operator="In",
                            values=required_zones,
                        )
                    ]
                )
            ]
        )

    if preferred_zones:
        preferred_zone_affinity = k8s_client.V1PreferredSchedulingTerm(
            weight=100,
            preference=k8s_client.V1NodeSelectorTerm(
                match_expressions=[
                    k8s_client.V1NodeSelectorRequirement(
                        key="topology.kubernetes.io/zone", operator="In", values=preferred_zones
                    )
                ]
            ),
        )

    return (required_zone_affinity, preferred_zone_affinity)


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
    required_zones: list[str] | None = None,
    preferred_zones: list[str] | None = None,
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

    required_affinity, preferred_affinity = _get_zone_affinities(required_zones, preferred_zones)
    affinity = k8s_client.V1Affinity(
        node_affinity=k8s_client.V1NodeAffinity(
            required_during_scheduling_ignored_during_execution=required_affinity,
            preferred_during_scheduling_ignored_during_execution=(
                [preferred_affinity] if preferred_affinity else None
            ),
        )
    )

    return get_pod_spec(
        name="zutils-worker",
        image=image,
        command=command,
        resources=resources,
        affinity=affinity,
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


def _wait_for_pod_start(pod_name: str, namespace: str, core_api: k8s_client.CoreV1Api):
    pod = core_api.read_namespaced_pod(name=pod_name, namespace=namespace)
    while pod.status.phase != "Running":
        logger.info(f"Waiting for `{pod_name}` to start before getting logs.")
        time.sleep(15)


def _reset_core_api(cluster_info: ClusterInfo):
    configuration, _ = get_cluster_data(cluster_info)
    k8s_client.Configuration.set_default(configuration)
    return k8s_client.CoreV1Api()


def stream_pod_logs(
    cluster_info: ClusterInfo,
    dep_selector: str | None = None,
    pod_name: str | None = None,
    namespace: str = "default",
    prefix: str = "",
    tail_lines: int | None = None,
):
    core_api = _reset_core_api(cluster_info)
    if pod_name is None:
        pods = core_api.list_namespaced_pod(namespace=namespace, label_selector=dep_selector).items
        if pods:
            pod_name = pods[0].metadata.name
        else:
            logger.info(f"No pods found for {dep_selector}. Waiting...")
            return

    _wait_for_pod_start(pod_name, namespace, core_api)
    try:
        log_stream = watch.Watch().stream(
            core_api.read_namespaced_pod_log,
            name=pod_name,
            container="main",
            namespace=namespace,
            tail_lines=tail_lines,
        )
        if tail_lines is None:
            for output in log_stream:
                logger.info(f"[{prefix}] {output}")
        else:
            result = []
            for output in log_stream:
                result.append(f"[{prefix}] {output}")
                if len(result) == tail_lines:
                    logger.info("\n".join(result))
                    result = []
            logger.info("\n".join(result))
    except ApiException as exc:
        if exc.status == 404:
            logger.info(f"Pod `{pod_name}` was deleted.")
            stream_pod_logs(
                cluster_info,
                dep_selector=dep_selector,
                pod_name=None,
                namespace=namespace,
                prefix=prefix,
                tail_lines=tail_lines,
            )
        else:
            logger.info(f"{exc.status}: {exc}")
            # resets credential config after timeout
            stream_pod_logs(
                cluster_info,
                dep_selector=dep_selector,
                pod_name=pod_name,
                namespace=namespace,
                prefix=prefix,
                tail_lines=tail_lines,
            )
