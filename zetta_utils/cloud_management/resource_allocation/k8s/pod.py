"""
Helpers for k8s pod.
"""

from __future__ import annotations

import functools
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Literal, Mapping, Optional

import kubernetes.client as k8s_client
from kubernetes.client.exceptions import ApiException

from kubernetes import config, watch  # type: ignore
from zetta_utils import log
from zetta_utils.cloud_management.resource_allocation import k8s

from .common import ClusterInfo, get_cluster_data, is_job_completed
from .secret import get_worker_env_vars
from .sidecar import (
    WAIT_FOR_CA_SCRIPT,
    build_main_proxy_envs,
    build_mproxy_sidecar,
    build_oom_sidecar,
    setup_mitmproxy_ca_volume,
)

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
    termination_grace_seconds: int = 60,
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

    sidecar_envs = list(envs)
    main_envs = build_main_proxy_envs(envs)

    host_aliases = host_aliases or []
    tolerations = tolerations or []
    volumes = list(volumes or [])
    volume_mounts = list(volume_mounts or [])
    main_volume_mounts, sidecar_volume_mounts = setup_mitmproxy_ca_volume(volumes, volume_mounts)

    log_pod_runtime_module = "zetta_utils.cloud_management.resource_allocation.k8s.log_pod_runtime"
    pre_stop_hook = k8s_client.V1Lifecycle(
        pre_stop=k8s_client.V1LifecycleHandler(
            _exec=k8s_client.V1ExecAction(
                command=["/bin/bash", "-c", f"python -m {log_pod_runtime_module}"]
            )
        )
    )

    main_container = k8s_client.V1Container(
        command=["/bin/bash", "-c"],
        args=[f"{WAIT_FOR_CA_SCRIPT} && {command}"],
        env=main_envs,
        name="main",
        image=image,
        image_pull_policy=image_pull_policy,
        lifecycle=pre_stop_hook,
        ports=[k8s_client.V1ContainerPort(container_port=29400)],
        resources=k8s_client.V1ResourceRequirements(
            limits=resources,
            requests=resource_requests or resources,
        ),
        termination_message_path="/dev/termination-log",
        termination_message_policy="File",
        volume_mounts=main_volume_mounts,
    )

    return k8s_client.V1PodSpec(
        affinity=affinity,
        containers=[
            main_container,
            build_oom_sidecar(image, sidecar_envs, volume_mounts),
            build_mproxy_sidecar(image, sidecar_envs, sidecar_volume_mounts),
        ],
        dns_policy=dns_policy,
        hostname=hostname,
        host_network=host_network,
        host_aliases=host_aliases,
        node_selector=node_selector,
        restart_policy=restart_policy,
        scheduler_name="default-scheduler",
        security_context={},
        subdomain=subdomain,
        termination_grace_period_seconds=termination_grace_seconds,
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
    restart_policy: Literal["Always", "Never", "OnFailure"] = "Always",
    gpu_accelerator_type: str | None = None,
    adc_available: bool = False,
    cave_secret_available: bool = False,
    required_zones: list[str] | None = None,
    preferred_zones: list[str] | None = None,
    worker_type: str | None = None,
    termination_grace_seconds: int = 300,
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
        termination_grace_seconds=termination_grace_seconds,
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


def _check_pod_disruption(pod, run_id, reported) -> str | None:
    """Check a pod event for disruptions. Returns a message string or None.

    Walks every container so a disruption in any sidecar (e.g. mproxy OOM)
    is reported distinctly. Previously the loop early-returned on the first
    terminated container, hiding others and omitting the container name.
    """
    pod_name = pod.metadata.name
    if pod_name in reported or run_id not in pod_name:
        return None
    node_name = pod.spec.node_name or "unknown"
    if pod.status.reason == "Evicted":
        reported.add(pod_name)
        return f"[Evicted] {pod_name} on node {node_name}."
    if not pod.status.container_statuses:
        return None
    msgs = []
    for cs in pod.status.container_statuses:
        state = cs.state
        if not (state and state.terminated):
            continue
        tag = None
        if state.terminated.reason == "OOMKilled":
            tag = "OOMKilled"
        elif state.terminated.exit_code == 137:
            tag = "SIGKILL"
        elif state.terminated.exit_code == 143:
            tag = "SIGTERM"
        if tag:
            msgs.append(f"[{tag}] {pod_name} container={cs.name} on node {node_name}.")
    if msgs:
        reported.add(pod_name)
        return "\n".join(msgs)
    return None


_RUN_EVENT_REASONS = {
    "Killing",
    "TaintManagerEviction",
    "ExceededGracePeriod",
    "Preempting",
    "Evicted",
}

# Container names whose K8s events we surface in events.log. Other containers
# (e.g. the `runtime` OOM-tracker) are filtered out to keep the signal clean.
_TRACKED_CONTAINER_EVENT_NAMES = ("container main", "container mproxy")


def _open_watcher_log(log_dir: str | None, run_id: str, filename: str) -> Path | None:
    if not log_dir:
        return None
    out_dir = Path(log_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename


def _append_watcher_log(log_path: Path | None, msg: str) -> None:
    if log_path is None:
        return
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def _load_core_v1_api() -> k8s_client.CoreV1Api:
    config.load_kube_config()
    return k8s_client.CoreV1Api()


def _resilient_watch(
    list_fn,
    on_event: Callable,
    *,
    namespace: str,
    description: str,
    stop_event: threading.Event | None = None,
    on_stream_end: Callable[[], None] | None = None,
) -> None:
    """Stream K8s objects via watch.Watch, retrying transient errors with backoff.

    Previously both watchers wrapped the `while` loop in a single try/except,
    so any `w.stream()` failure (e.g. a momentary GKE master unreachable at
    startup) permanently killed disruption detection for the rest of the run.
    Here the try/except is inside the loop and resets backoff on clean cycles.
    """
    w = watch.Watch()
    backoff = 1.0
    try:
        while not (stop_event and stop_event.is_set()):
            try:
                for event in w.stream(list_fn, namespace=namespace, timeout_seconds=30):
                    if stop_event and stop_event.is_set():
                        break
                    on_event(event["object"])
                if on_stream_end is not None:
                    on_stream_end()
                backoff = 1.0
            except Exception as err:  # pylint: disable=broad-exception-caught
                logger.warning(f"{description} transient error, retrying in {backoff:.1f}s: {err}")
                if stop_event and stop_event.wait(backoff):
                    break
                backoff = min(backoff * 2, 30.0)
    finally:
        w.stop()


def watch_for_pod_disruptions(
    run_id: str,
    namespace="default",
    log_dir: str | None = None,
    stop_event: threading.Event | None = None,
):
    log_path = _open_watcher_log(log_dir, run_id, "disruptions.log")
    v1 = _load_core_v1_api()
    reported: set[str] = set()

    def on_event(pod):
        msg = _check_pod_disruption(pod, run_id, reported)
        if msg:
            logger.warning(msg)
            _append_watcher_log(log_path, msg)

    _resilient_watch(
        v1.list_namespaced_pod,
        on_event,
        namespace=namespace,
        description="Pod disruption watcher",
        stop_event=stop_event,
    )


def watch_for_run_events(
    run_id: str,
    namespace="default",
    log_dir: str | None = None,
    stop_event: threading.Event | None = None,
):
    log_path = _open_watcher_log(log_dir, run_id, "events.log")
    v1 = _load_core_v1_api()
    pending: list[str] = []

    def on_event(obj):
        pod_name = obj.involved_object.name if obj.involved_object else ""
        if run_id not in pod_name or obj.reason not in _RUN_EVENT_REASONS:
            return
        message = obj.message or ""
        if "container " in message and not any(
            n in message for n in _TRACKED_CONTAINER_EVENT_NAMES
        ):
            return
        msg = f"[K8s:{obj.reason}] {pod_name}: {message}"
        pending.append(msg)
        _append_watcher_log(log_path, msg)

    def flush():
        if pending:
            logger.warning("\n".join(pending))
            pending.clear()

    _resilient_watch(
        v1.list_namespaced_event,
        on_event,
        namespace=namespace,
        description="Run event watcher",
        stop_event=stop_event,
        on_stream_end=flush,
    )


_TRIGGERED_SCALE_UP_MIG_PATTERN = re.compile(r"\{([^ ]+) \d+->\d+ \(max: \d+\)\}")


def watch_for_triggered_scale_up(
    name_prefix: str,
    on_event: Callable[[str, list[str]], None],
    namespace: str = "default",
    stop_event: threading.Event | None = None,
) -> None:
    """Watch ``TriggeredScaleUp`` events from the cluster autoscaler.

    The cluster autoscaler emits this event on a Pod when it decides to
    scale up to make space for the Pod, with message format
    ``"pod triggered scale-up: [{MIG_NAME N->M (max: K)}, ...]"``. Server-side
    filters by ``reason=TriggeredScaleUp`` and ``involvedObject.kind=Pod``;
    client-side filters by ``name_prefix`` on the involved Pod's name.
    Calls ``on_event(pod_name, mig_names)`` with the list of attempted MIGs.
    """
    v1 = _load_core_v1_api()

    def _handle(obj):
        pod_name = obj.involved_object.name if obj.involved_object else ""
        if not pod_name.startswith(name_prefix):
            return
        mig_names = _TRIGGERED_SCALE_UP_MIG_PATTERN.findall(obj.message or "")
        if mig_names:
            on_event(pod_name, mig_names)

    _resilient_watch(
        functools.partial(
            v1.list_namespaced_event,
            field_selector="reason=TriggeredScaleUp,involvedObject.kind=Pod",
        ),
        _handle,
        namespace=namespace,
        description="TriggeredScaleUp watcher",
        stop_event=stop_event,
    )
