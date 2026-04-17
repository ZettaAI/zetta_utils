"""
Worker-pod sidecars and the main-container wiring they drive.

Two sidecars run alongside `main`:
- `runtime` (oom_tracker): captures OOMs / exit signals for `main`.
- `mproxy` (gcs_tracker / mitmproxy): intercepts `storage.googleapis.com`
  traffic for GCS request + egress accounting. Because HTTPS_PROXY is set
  on `main` to route through mproxy, this module also owns the env-var,
  CA-bundle, and `wait_for_ca` plumbing on the main container side.
"""

from __future__ import annotations

from typing import Iterable

from kubernetes import client as k8s_client

# Shared emptyDir mount for mitmproxy CA cert (written by mproxy, read by main).
_CA_VOLUME_NAME = "mitmproxy-ca"
_CA_MOUNT_PATH = "/tmp/mitmproxy-ca"
_CA_BUNDLE_PATH = f"{_CA_MOUNT_PATH}/combined-ca-bundle.pem"
_READY_MARKER = f"{_CA_MOUNT_PATH}/ready"

# Env vars that various HTTP clients consult for the CA bundle. The combined
# bundle is created in the wait_for_ca script (system CAs + mitmproxy CA).
_CA_ENV_VARS = (
    "REQUESTS_CA_BUNDLE",  # Python requests
    "SSL_CERT_FILE",  # aiohttp (gcsfs), general OpenSSL
    "CURL_CA_BUNDLE",  # curl / libcurl
    "HTTPLIB2_CA_CERTS",  # httplib2 (some google-cloud libs)
    "TENSORSTORE_CA_BUNDLE",  # tensorstore
    "GRPC_DEFAULT_SSL_ROOTS_FILE_PATH",  # gRPC
)

# NO_PROXY excludes that bypass mproxy:
# - 127.0.0.1 / kubernetes.default.svc / metadata.google.internal: infra
# - amazonaws.com: AWS (SQS etc.) must not depend on mproxy being healthy
_NO_PROXY_HOSTS = ",".join(
    [
        "127.0.0.1",
        "metadata.google.internal",
        "kubernetes.default.svc",
        "amazonaws.com",
        ".amazonaws.com",
    ]
)


def _build_wait_for_ca_script() -> str:
    """Bash preamble for the main container.

    Blocks until mproxy signals readiness by creating the ready marker.
    If the marker contains "disabled" (mitmdump not installed), unsets the
    proxy env vars so main runs direct. Otherwise concatenates system CAs +
    mitmproxy CA into the combined bundle and exports CA-bundle env vars
    for every HTTP client family.
    """
    unset_vars = " ".join(
        ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "NO_PROXY", "no_proxy")
        + _CA_ENV_VARS
    )
    exports = " && ".join(f"export {v}={_CA_BUNDLE_PATH}" for v in _CA_ENV_VARS)
    return (
        "echo 'Waiting for GCS tracker ready signal...' && "
        f"while [ ! -f {_READY_MARKER} ]; do sleep 0.5; done && "
        f"if grep -q 'disabled' {_READY_MARKER} 2>/dev/null; then "
        "echo 'GCS tracking disabled, unsetting proxy vars' && "
        f"unset {unset_vars}; "
        "else "
        "echo 'GCS tracking enabled, creating combined CA bundle' && "
        f"cat /etc/ssl/certs/ca-certificates.crt {_CA_MOUNT_PATH}/mitmproxy-ca-cert.pem "
        f"> {_CA_BUNDLE_PATH} && "
        f"{exports}; "
        "fi"
    )


WAIT_FOR_CA_SCRIPT = _build_wait_for_ca_script()


def build_main_proxy_envs(base_envs: list[k8s_client.V1EnvVar]) -> list[k8s_client.V1EnvVar]:
    """Return a new env list with the main container's proxy + CA-bundle vars appended.

    Main routes through mproxy on localhost:8080. NO_PROXY excludes AWS so
    SQS/etc. don't depend on mproxy being healthy. CA-bundle vars point at
    the combined bundle that wait_for_ca builds.
    """
    envs = list(base_envs)
    envs.extend(
        [
            k8s_client.V1EnvVar(name="HTTPS_PROXY", value="http://localhost:8080"),
            k8s_client.V1EnvVar(name="HTTP_PROXY", value="http://localhost:8080"),
            k8s_client.V1EnvVar(name="https_proxy", value="http://localhost:8080"),
            k8s_client.V1EnvVar(name="http_proxy", value="http://localhost:8080"),
            k8s_client.V1EnvVar(name="NO_PROXY", value=_NO_PROXY_HOSTS),
            k8s_client.V1EnvVar(name="no_proxy", value=_NO_PROXY_HOSTS),
        ]
    )
    for name in _CA_ENV_VARS:
        envs.append(k8s_client.V1EnvVar(name=name, value=_CA_BUNDLE_PATH))
    return envs


def setup_mitmproxy_ca_volume(
    volumes: list[k8s_client.V1Volume],
    base_volume_mounts: list[k8s_client.V1VolumeMount],
) -> tuple[list[k8s_client.V1VolumeMount], list[k8s_client.V1VolumeMount]]:
    """Append the shared CA emptyDir to `volumes` and return (main_mounts, sidecar_mounts).

    Both main and mproxy mount the same emptyDir — mproxy writes the CA cert,
    main reads it (via wait_for_ca). Two identical lists are returned so
    callers can augment one without affecting the other.
    """
    volumes.append(
        k8s_client.V1Volume(
            name=_CA_VOLUME_NAME,
            empty_dir=k8s_client.V1EmptyDirVolumeSource(),
        )
    )
    ca_mount = k8s_client.V1VolumeMount(name=_CA_VOLUME_NAME, mount_path=_CA_MOUNT_PATH)
    return list(base_volume_mounts) + [ca_mount], list(base_volume_mounts) + [ca_mount]


def _build_sidecar_container(
    *,
    name: str,
    module: str,
    image: str,
    envs: list[k8s_client.V1EnvVar],
    volume_mounts: list[k8s_client.V1VolumeMount],
    resources: k8s_client.V1ResourceRequirements,
    ports: Iterable[k8s_client.V1ContainerPort] = (),
    startup_probe: k8s_client.V1Probe | None = None,
    liveness_probe: k8s_client.V1Probe | None = None,
) -> k8s_client.V1Container:
    """Shared V1Container construction for `python -m <module>` sidecars."""
    return k8s_client.V1Container(
        command=["/bin/bash", "-c"],
        args=[f"python -m {module}"],
        env=envs,
        name=name,
        image=image,
        termination_message_path="/dev/termination-log",
        termination_message_policy="File",
        volume_mounts=volume_mounts,
        ports=list(ports),
        resources=resources,
        startup_probe=startup_probe,
        liveness_probe=liveness_probe,
    )


def build_oom_sidecar(
    image: str,
    envs: list[k8s_client.V1EnvVar],
    volume_mounts: list[k8s_client.V1VolumeMount],
) -> k8s_client.V1Container:
    """OOM tracker sidecar.

    Gets small CPU/memory requests so it isn't starved under node pressure —
    exactly when main-container OOMs happen and this tracker needs to observe them.
    """
    return _build_sidecar_container(
        name="runtime",
        module="zetta_utils.cloud_management.resource_allocation.k8s.oom_tracker",
        image=image,
        envs=envs,
        volume_mounts=volume_mounts,
        resources=k8s_client.V1ResourceRequirements(
            requests={"cpu": "50m", "memory": "128Mi", "ephemeral-storage": "100Mi"},
            limits={"memory": "256Mi"},
        ),
    )


def build_mproxy_sidecar(
    image: str,
    envs: list[k8s_client.V1EnvVar],
    volume_mounts: list[k8s_client.V1VolumeMount],
) -> k8s_client.V1Container:
    """GCS tracker sidecar (mitmproxy) intercepting storage.googleapis.com.

    Without a CPU request, mproxy was starved during main-container CPU bursts
    and its TLS handshakes deadlocked, producing ProxyError on main's GCS calls.
    Memory request+limit turns OOM into a clean container-level restart instead
    of the node-level OOM-killer picking a victim. TCP probes detect hangs; the
    startup probe grants CA-cert generation time before port 8080 binds.
    """
    return _build_sidecar_container(
        name="mproxy",
        module="zetta_utils.cloud_management.resource_allocation.k8s.gcs_tracker",
        image=image,
        envs=envs,
        volume_mounts=volume_mounts,
        ports=[k8s_client.V1ContainerPort(container_port=8080, name="proxy")],
        resources=k8s_client.V1ResourceRequirements(
            requests={"cpu": "200m", "memory": "512Mi", "ephemeral-storage": "100Mi"},
            limits={"memory": "1Gi"},
        ),
        startup_probe=k8s_client.V1Probe(
            tcp_socket=k8s_client.V1TCPSocketAction(port=8080),
            period_seconds=2,
            failure_threshold=30,
        ),
        liveness_probe=k8s_client.V1Probe(
            tcp_socket=k8s_client.V1TCPSocketAction(port=8080),
            period_seconds=10,
            failure_threshold=3,
            timeout_seconds=3,
        ),
    )
