"""
Helpers for k8s pod.
"""

from __future__ import annotations

import time
from typing import Dict, List, Literal, Mapping, Optional

from kubernetes.client.exceptions import ApiException

from kubernetes import client as k8s_client
from kubernetes import config, watch  # type: ignore
from zetta_utils import log
from zetta_utils.cloud_management.resource_allocation import k8s

from .common import ClusterInfo, get_cluster_data
from .secret import get_worker_env_vars

logger = log.get_logger("zetta_utils")


def get_pod_spec(  # pylint: disable=too-many-locals
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

    envs.extend(
        [
            k8s_client.V1EnvVar(
                name="POD_NAME",
                value_from=k8s_client.V1EnvVarSource(
                    field_ref=k8s_client.V1ObjectFieldSelector(field_path="metadata.name")
                ),
            ),
            k8s_client.V1EnvVar(
                name="NODE_NAME",
                value_from=k8s_client.V1EnvVarSource(
                    field_ref=k8s_client.V1ObjectFieldSelector(field_path="spec.nodeName")
                ),
            ),
        ]
    )
    envs.extend(get_worker_env_vars(env_secret_mapping))

    # Base envs for sidecars (without proxy)
    sidecar_envs = list(envs)

    # Configure proxy for GCS traffic tracking via sidecar (main container only)
    # The gcs-traffic sidecar runs mitmproxy on port 8080
    # Set both uppercase (standard) and lowercase (tensorstore/libcurl) proxy vars
    main_envs = list(envs)
    main_envs.extend(
        [
            k8s_client.V1EnvVar(name="HTTPS_PROXY", value="http://localhost:8080"),
            k8s_client.V1EnvVar(name="HTTP_PROXY", value="http://localhost:8080"),
            k8s_client.V1EnvVar(name="https_proxy", value="http://localhost:8080"),
            k8s_client.V1EnvVar(name="http_proxy", value="http://localhost:8080"),
            # Ensure GCS traffic goes through proxy (don't bypass for localhost)
            k8s_client.V1EnvVar(name="NO_PROXY", value="127.0.0.1,metadata.google.internal"),
            k8s_client.V1EnvVar(name="no_proxy", value="127.0.0.1,metadata.google.internal"),
        ]
    )

    host_aliases = host_aliases or []
    tolerations = tolerations or []
    volumes = list(volumes or [])
    volume_mounts = list(volume_mounts or [])

    # Add shared volume for mitmproxy CA certificate
    mitmproxy_ca_volume = k8s_client.V1Volume(
        name="mitmproxy-ca",
        empty_dir=k8s_client.V1EmptyDirVolumeSource(),
    )
    volumes.append(mitmproxy_ca_volume)

    mitmproxy_ca_mount = k8s_client.V1VolumeMount(
        name="mitmproxy-ca",
        mount_path="/tmp/mitmproxy-ca",
    )

    # Main container mounts
    main_volume_mounts = list(volume_mounts)
    main_volume_mounts.append(mitmproxy_ca_mount)

    # Sidecar mounts (gcs-traffic needs the CA dir to write to)
    sidecar_volume_mounts = list(volume_mounts)
    sidecar_volume_mounts.append(mitmproxy_ca_mount)

    # Add CA bundle env vars for main container to trust mitmproxy CA
    # Different libraries use different env vars for CA certificates:
    # - REQUESTS_CA_BUNDLE: Python requests library
    # - SSL_CERT_FILE: aiohttp (used by gcsfs), general OpenSSL
    # - CURL_CA_BUNDLE: curl and libraries using libcurl
    # - HTTPLIB2_CA_CERTS: httplib2 (used by some google-cloud libs)
    # - TENSORSTORE_CA_BUNDLE: tensorstore (uses libcurl internally)
    # The combined bundle is created in wait_for_ca script (system CAs + mitmproxy CA)
    ca_cert_path = "/tmp/mitmproxy-ca/combined-ca-bundle.pem"
    ca_env_vars = [
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_FILE",
        "CURL_CA_BUNDLE",
        "HTTPLIB2_CA_CERTS",
        "TENSORSTORE_CA_BUNDLE",
    ]
    for ca_env_var in ca_env_vars:
        main_envs.append(k8s_client.V1EnvVar(name=ca_env_var, value=ca_cert_path))

    module = "zetta_utils.cloud_management.resource_allocation.k8s.log_pod_runtime"
    cmd = f"python -m {module}"
    pre_stop_hook = k8s_client.V1Lifecycle(
        pre_stop=k8s_client.V1LifecycleHandler(
            _exec=k8s_client.V1ExecAction(command=["/bin/bash", "-c", cmd])
        )
    )

    # Wait for mitmproxy CA cert before running main command
    # If tracking is disabled (mitmproxy not installed), unset proxy vars
    # If enabled, create combined CA bundle (system CAs + mitmproxy CA)
    wait_for_ca = (
        "echo 'Waiting for GCS tracker ready signal...' && "
        "while [ ! -f /tmp/mitmproxy-ca/ready ]; do sleep 0.5; done && "
        "if grep -q 'disabled' /tmp/mitmproxy-ca/ready 2>/dev/null; then "
        "echo 'GCS tracking disabled, unsetting proxy vars' && "
        "unset HTTPS_PROXY HTTP_PROXY https_proxy http_proxy NO_PROXY no_proxy "
        "REQUESTS_CA_BUNDLE SSL_CERT_FILE CURL_CA_BUNDLE HTTPLIB2_CA_CERTS TENSORSTORE_CA_BUNDLE; "
        "else "
        "echo 'GCS tracking enabled, creating combined CA bundle' && "
        "cat /etc/ssl/certs/ca-certificates.crt "
        "/tmp/mitmproxy-ca/mitmproxy-ca-cert.pem "
        "> /tmp/mitmproxy-ca/combined-ca-bundle.pem && "
        "export REQUESTS_CA_BUNDLE=/tmp/mitmproxy-ca/combined-ca-bundle.pem && "
        "export SSL_CERT_FILE=/tmp/mitmproxy-ca/combined-ca-bundle.pem && "
        "export CURL_CA_BUNDLE=/tmp/mitmproxy-ca/combined-ca-bundle.pem && "
        "export HTTPLIB2_CA_CERTS=/tmp/mitmproxy-ca/combined-ca-bundle.pem && "
        "export TENSORSTORE_CA_BUNDLE=/tmp/mitmproxy-ca/combined-ca-bundle.pem; "
        "fi"
    )
    wrapped_command = f"{wait_for_ca} && {command}"

    ports = [k8s_client.V1ContainerPort(container_port=29400)]
    container = k8s_client.V1Container(
        command=["/bin/bash", "-c"],
        args=[wrapped_command],
        env=main_envs,
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
        volume_mounts=main_volume_mounts,
    )

    module = "zetta_utils.cloud_management.resource_allocation.k8s.oom_tracker"
    oom_sidecar = k8s_client.V1Container(
        command=["/bin/bash", "-c"],
        args=[f"python -m {module}"],
        env=sidecar_envs,
        name="runtime",
        image=image,
        termination_message_path="/dev/termination-log",
        termination_message_policy="File",
        volume_mounts=volume_mounts,
        resources=k8s_client.V1ResourceRequirements(
            requests={"ephemeral-storage": "100Mi"},
        ),
    )

    # GCS tracker sidecar using mitmproxy for request classification
    gcs_tracker_module = "zetta_utils.cloud_management.resource_allocation.k8s.gcs_tracker"
    gcs_traffic_sidecar = k8s_client.V1Container(
        command=["/bin/bash", "-c"],
        args=[f"python -m {gcs_tracker_module}"],
        env=sidecar_envs,
        name="mproxy",
        image=image,
        termination_message_path="/dev/termination-log",
        termination_message_policy="File",
        volume_mounts=sidecar_volume_mounts,
        ports=[k8s_client.V1ContainerPort(container_port=8080, name="proxy")],
        resources=k8s_client.V1ResourceRequirements(
            requests={"ephemeral-storage": "100Mi"},
        ),
    )

    return k8s_client.V1PodSpec(
        affinity=affinity,
        containers=[container, oom_sidecar, gcs_traffic_sidecar],
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


def get_zone_affinities(
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

    required_affinity, preferred_affinity = get_zone_affinities(required_zones, preferred_zones)
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


def _wait_for_pod_start(
    pod_name: str,
    namespace: str,
    core_api: k8s_client.CoreV1Api,
    poll_interval: int = 15,
    timeout: int = 600,
):
    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise TimeoutError(
                f"Timed out waiting for pod `{pod_name}` to become ready after {timeout} seconds."
            )

        try:
            pod = core_api.read_namespaced_pod(name=pod_name, namespace=namespace)
        except k8s_client.exceptions.ApiException as e:
            if e.status == 404:
                logger.info(f"Pod `{pod_name}` not found yet. Retrying...")
                time.sleep(poll_interval)
                continue
            raise

        phase = pod.status.phase
        container_statuses = pod.status.container_statuses or []

        if container_statuses and all(cs.ready for cs in container_statuses):
            logger.info(f"Pod `{pod_name}` is running and all containers are ready.")
            return

        logger.info(
            f"Waiting for `{pod_name}` to start. "
            f"Phase: {phase}. Ready states: {[cs.ready for cs in container_statuses]}"
        )
        time.sleep(poll_interval)


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


def watch_for_oom_kills(run_id: str, namespace="default"):
    config.load_kube_config()
    v1 = k8s_client.CoreV1Api()
    w = watch.Watch()
    pods = set()
    try:
        for event in w.stream(v1.list_namespaced_pod, namespace=namespace):
            pod = event["object"]
            pod_name = pod.metadata.name
            if pod_name in pods or run_id not in pod_name:
                continue
            if pod.status.container_statuses:
                for cs in pod.status.container_statuses:
                    state = cs.state
                    if state and state.terminated and state.terminated.reason == "OOMKilled":
                        logger.warning(f"⚠️ [OOMKilled] {pod_name}.")
                        pods.add(pod_name)
    except Exception as err:  # pylint: disable=broad-exception-caught
        logger.warning(err)
    finally:
        w.stop()
