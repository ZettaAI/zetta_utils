"""
Helpers for k8s pod.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Literal, Mapping, Optional

from kubernetes.client.exceptions import ApiException

from kubernetes import client as k8s_client
from kubernetes import config, watch  # type: ignore
from zetta_utils import log
from zetta_utils.cloud_management.resource_allocation import k8s

from .common import ClusterInfo, get_cluster_data, is_job_completed
from .secret import get_worker_env_vars

logger = log.get_logger("zetta_utils")
simple_logger = log.get_simple_logger("zetta_utils.fwd")


class PodTerminatedError(Exception):
    pass


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
    subdomain: Optional[str] = None,
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
            k8s_client.V1EnvVar(
                name="NO_PROXY", value="127.0.0.1,metadata.google.internal,kubernetes.default.svc"
            ),
            k8s_client.V1EnvVar(
                name="no_proxy", value="127.0.0.1,metadata.google.internal,kubernetes.default.svc"
            ),
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
        "GRPC_DEFAULT_SSL_ROOTS_FILE_PATH",
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
        "REQUESTS_CA_BUNDLE SSL_CERT_FILE CURL_CA_BUNDLE HTTPLIB2_CA_CERTS TENSORSTORE_CA_BUNDLE "
        "GRPC_DEFAULT_SSL_ROOTS_FILE_PATH; "
        "else "
        "echo 'GCS tracking enabled, creating combined CA bundle' && "
        "cat /etc/ssl/certs/ca-certificates.crt "
        "/tmp/mitmproxy-ca/mitmproxy-ca-cert.pem "
        "> /tmp/mitmproxy-ca/combined-ca-bundle.pem && "
        "export REQUESTS_CA_BUNDLE=/tmp/mitmproxy-ca/combined-ca-bundle.pem && "
        "export SSL_CERT_FILE=/tmp/mitmproxy-ca/combined-ca-bundle.pem && "
        "export CURL_CA_BUNDLE=/tmp/mitmproxy-ca/combined-ca-bundle.pem && "
        "export HTTPLIB2_CA_CERTS=/tmp/mitmproxy-ca/combined-ca-bundle.pem && "
        "export TENSORSTORE_CA_BUNDLE=/tmp/mitmproxy-ca/combined-ca-bundle.pem && "
        "export GRPC_DEFAULT_SSL_ROOTS_FILE_PATH=/tmp/mitmproxy-ca/combined-ca-bundle.pem; "
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
        subdomain=subdomain,
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
    worker_type: str | None = None,
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
    if worker_type:
        envs.append(k8s_client.V1EnvVar(name="WORKER_TYPE", value=worker_type))

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
                logger.debug(f"Pod `{pod_name}` not found yet. Retrying...")
                time.sleep(poll_interval)
                continue
            raise

        phase = pod.status.phase
        if phase in ("Succeeded", "Failed"):
            raise PodTerminatedError(f"Pod `{pod_name}` in terminal phase: {phase}")
        container_statuses = pod.status.container_statuses or []

        if container_statuses and all(cs.ready for cs in container_statuses):
            logger.debug(f"Pod `{pod_name}` is running and all containers are ready.")
            return

        logger.debug(
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
                simple_logger.info(f"[{prefix}] {output}")
        else:
            result = []
            for output in log_stream:
                result.append(f"[{prefix}] {output}")
                if len(result) == tail_lines:
                    simple_logger.info("\n".join(result))
                    result = []
            simple_logger.info("\n".join(result))
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


def follow_rank0_logs(
    run_id: str,
    cluster_info: ClusterInfo,
    namespace: str = "default",
    tail_lines: int | None = None,
) -> str | None:
    selector = f"run-id={run_id},training-rank=0"
    logger.info(f"Waiting for rank 0 pod (selector: {selector})...")
    while True:
        core_api = _reset_core_api(cluster_info)
        pods = core_api.list_namespaced_pod(namespace=namespace, label_selector=selector).items
        if pods:
            pod_name = pods[0].metadata.name
            logger.info(f"Streaming logs from pod: {pod_name}")
            try:
                stream_pod_logs(
                    cluster_info,
                    pod_name=pod_name,
                    namespace=namespace,
                    prefix=f"rank0:{pod_name[-7:]}",
                    tail_lines=tail_lines,
                )
                logger.info("Log stream ended, polling for rank 0...")
            except PodTerminatedError:
                logger.info("Rank 0 pod terminated, checking job status...")
        job_status = is_job_completed(f"run-{run_id}", cluster_info, namespace)
        if job_status is not None:
            if job_status == "succeeded":
                logger.info(f"Master job for run `{run_id}` succeeded. Stopping log follow.")
            else:
                logger.error(f"Master job for run `{run_id}` {job_status}. Stopping log follow.")
            return job_status

        time.sleep(5)


def _capture_single_pod_logs(
    cluster_info: ClusterInfo,
    pod_name: str,
    log_file: Path,
    container: str = "main",
    namespace: str = "default",
    stop_event: threading.Event | None = None,
):
    core_api = _reset_core_api(cluster_info)
    try:
        _wait_for_pod_start(pod_name, namespace, core_api)
    except PodTerminatedError:
        logger.debug(f"Pod `{pod_name}` terminated before log capture started.")
        return
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning(f"Failed waiting for pod `{pod_name}` to start: {exc}")
        return
    try:
        resp = core_api.read_namespaced_pod_log(
            name=pod_name,
            container=container,
            namespace=namespace,
            follow=True,
            _preload_content=False,
        )
        with open(log_file, "a", encoding="utf-8") as f:
            for line in resp.stream(decode_content=True):
                if stop_event and stop_event.is_set():
                    break
                f.write(line.decode("utf-8", errors="replace"))
        resp.release_conn()
    except ApiException as exc:
        if exc.status in (404, 410):
            logger.debug(f"Pod `{pod_name}` gone ({exc.status}), stopping log capture.")
        else:
            logger.debug(f"Log stream ended for `{pod_name}`: {exc.reason}")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning(f"Error capturing logs for `{pod_name}`: {exc}")


def capture_pod_logs(
    run_id: str,
    cluster_info: ClusterInfo,
    log_dir: str,
    namespace: str = "default",
    stop_event: threading.Event | None = None,
):
    out_dir = Path(log_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Capturing pod logs to {out_dir}")
    seen: set[str] = set()
    selector = f"run-id={run_id}"
    executor = ThreadPoolExecutor(max_workers=32)
    while True:
        if stop_event and stop_event.is_set():
            logger.info(f"Stop requested, cancelling log capture for {out_dir}")
            executor.shutdown(wait=False, cancel_futures=True)
            return
        try:
            core_api = _reset_core_api(cluster_info)
            pods = core_api.list_namespaced_pod(namespace=namespace, label_selector=selector).items
            for pod in pods:
                pod_name = pod.metadata.name
                if pod_name in seen:
                    continue
                seen.add(pod_name)
                log_file = out_dir / f"{pod_name}.log"
                executor.submit(
                    _capture_single_pod_logs,
                    cluster_info,
                    pod_name,
                    log_file,
                    "main",
                    namespace,
                    stop_event,
                )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning(f"Error listing pods for log capture: {exc}")
        job_status = is_job_completed(f"run-{run_id}", cluster_info, namespace)
        if job_status is not None:
            logger.info(f"Job {job_status}, flushing captured logs in {out_dir}")
            executor.shutdown(wait=True, cancel_futures=True)
            return
        time.sleep(10)


def get_pod_postmortem(
    run_id: str,
    cluster_info: ClusterInfo,
    log_dir: str = "logs",
    namespace: str = "default",
):
    """Query K8s for master pod crash info and write postmortem.log."""
    try:
        core_api = _reset_core_api(cluster_info)
        job_name = f"run-{run_id}"
        pods = core_api.list_namespaced_pod(
            namespace=namespace, label_selector=f"job-name={job_name}"
        ).items
        if not pods:
            logger.error(f"No master pod found for job `{job_name}` — already deleted?")
            return

        out_dir = Path(log_dir) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        postmortem_path = out_dir / "postmortem.log"

        lines: list[str] = []
        for pod in pods:
            pod_name = pod.metadata.name
            node_name = pod.spec.node_name or "unknown"
            phase = pod.status.phase or "unknown"

            lines.append(f"=== Pod: {pod_name} ===")
            lines.append(f"Node: {node_name}")
            lines.append(f"Phase: {phase}")

            for cs in pod.status.container_statuses or []:
                terminated = cs.state.terminated if cs.state else None
                if terminated:
                    lines.append(
                        f"Container `{cs.name}`: reason={terminated.reason}, "
                        f"exit_code={terminated.exit_code}, "
                        f"message={terminated.message}"
                    )
                elif cs.state and cs.state.waiting:
                    lines.append(
                        f"Container `{cs.name}`: waiting, reason={cs.state.waiting.reason}"
                    )

            try:
                events = core_api.list_namespaced_event(
                    namespace=namespace,
                    field_selector=f"involvedObject.name={pod_name}",
                ).items
                if events:
                    lines.append("Events:")
                    for evt in sorted(
                        events, key=lambda e: e.last_timestamp or e.event_time or ""
                    ):
                        ts = evt.last_timestamp or evt.event_time or "?"
                        lines.append(f"  {ts} [{evt.reason}] {evt.message}")
            except Exception as exc:  # pylint: disable=broad-exception-caught
                lines.append(f"Failed to fetch events: {exc}")

            lines.append("")

        postmortem_text = "\n".join(lines)
        with open(postmortem_path, "w", encoding="utf-8") as f:
            f.write(postmortem_text)

        logger.error(f"Post-mortem for run `{run_id}`:\n{postmortem_text}")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error(f"Failed to collect post-mortem for run `{run_id}`: {exc}")


def _check_pod_disruption(pod, run_id, reported):  # pylint: disable=too-many-return-statements
    """Check a pod event for disruptions. Returns a message string or None."""
    pod_name = pod.metadata.name
    if pod_name in reported or run_id not in pod_name:
        return None
    node_name = pod.spec.node_name or "unknown"
    if pod.status.reason == "Evicted":
        reported.add(pod_name)
        return f"[Evicted] {pod_name} on node {node_name}."
    if not pod.status.container_statuses:
        return None
    for cs in pod.status.container_statuses:
        state = cs.state
        if not (state and state.terminated):
            continue
        msg = None
        if state.terminated.reason == "OOMKilled":
            msg = f"[OOMKilled] {pod_name} on node {node_name}."
        elif state.terminated.exit_code == 137:
            msg = f"[SIGKILL] {pod_name} on node {node_name}."
        elif state.terminated.exit_code == 143:
            msg = f"[SIGTERM] {pod_name} on node {node_name}."
        if msg:
            reported.add(pod_name)
        return msg
    return None


def watch_for_pod_disruptions(
    run_id: str,
    namespace="default",
    log_dir: str | None = None,
    stop_event: threading.Event | None = None,
):
    log_path = None
    if log_dir:
        out_dir = Path(log_dir) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "disruptions.log"
    config.load_kube_config()
    v1 = k8s_client.CoreV1Api()
    w = watch.Watch()
    reported: set[str] = set()
    try:
        while not (stop_event and stop_event.is_set()):
            for event in w.stream(v1.list_namespaced_pod, namespace=namespace, timeout_seconds=30):
                if stop_event and stop_event.is_set():
                    break
                msg = _check_pod_disruption(event["object"], run_id, reported)
                if msg:
                    logger.warning(msg)
                    if log_path:
                        with open(log_path, "a", encoding="utf-8") as f:
                            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    except Exception as err:  # pylint: disable=broad-exception-caught
        logger.warning(err)
    finally:
        w.stop()


_RUN_EVENT_REASONS = {
    "Killing",
    "TaintManagerEviction",
    "ExceededGracePeriod",
    "Preempting",
    "Evicted",
}


def watch_for_run_events(
    run_id: str,
    namespace="default",
    log_dir: str | None = None,
    stop_event: threading.Event | None = None,
):
    log_path = None
    if log_dir:
        out_dir = Path(log_dir) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "events.log"
    config.load_kube_config()
    v1 = k8s_client.CoreV1Api()
    w = watch.Watch()
    try:
        while not (stop_event and stop_event.is_set()):
            pending: list[str] = []
            for event in w.stream(
                v1.list_namespaced_event, namespace=namespace, timeout_seconds=30
            ):
                if stop_event and stop_event.is_set():
                    break
                obj = event["object"]
                pod_name = obj.involved_object.name if obj.involved_object else ""
                if run_id not in pod_name:
                    continue
                if obj.reason in _RUN_EVENT_REASONS:
                    if "container " in (obj.message or "") and "container main" not in (
                        obj.message or ""
                    ):
                        continue
                    msg = f"[K8s:{obj.reason}] {pod_name}: {obj.message}"
                    pending.append(msg)
                    if log_path:
                        with open(log_path, "a", encoding="utf-8") as f:
                            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
            if pending:
                logger.warning("\n".join(pending))
    except Exception as err:  # pylint: disable=broad-exception-caught
        logger.warning(f"Event watcher error: {err}")
    finally:
        w.stop()
