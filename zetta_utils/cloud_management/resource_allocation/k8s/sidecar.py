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


# How long main will wait for mproxy to signal ready before failing the pod.
# GCS tracking is a hard requirement (for customer billing), so there is no
# fall-back-to-direct path; a persistent mproxy failure manifests as a
# visible pod-level crash-loop via restart_policy=Always.
_WAIT_FOR_READY_SECONDS = 120


def _build_wait_for_ca_script() -> str:
    """Bash preamble for the main container.

    Waits up to `_WAIT_FOR_READY_SECONDS` for mproxy to write the ready
    marker. On timeout, exits 1 so main never starts without tracking.
    On success, builds the combined system+mitmproxy CA bundle and exports
    the CA-bundle env vars for every HTTP client family.
    """
    exports = "\n".join(f"export {v}={_CA_BUNDLE_PATH}" for v in _CA_ENV_VARS)
    iters = _WAIT_FOR_READY_SECONDS * 2
    return f"""
set -e
echo 'Waiting for GCS tracker ready signal (up to {_WAIT_FOR_READY_SECONDS}s)...'
for _ in $(seq 1 {iters}); do
    if [ -f {_READY_MARKER} ]; then break; fi
    sleep 0.5
done
if [ ! -f {_READY_MARKER} ]; then
    echo 'GCS tracker did not signal ready within {_WAIT_FOR_READY_SECONDS}s; failing pod' >&2
    exit 1
fi
echo 'GCS tracking enabled, creating combined CA bundle'
cat /etc/ssl/certs/ca-certificates.crt {_CA_MOUNT_PATH}/mitmproxy-ca-cert.pem > {_CA_BUNDLE_PATH}
{exports}
""".strip()


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
            # Python + zetta_utils.run import chain (firestore client, etc.)
            # is ~200MB at rest before this process does any work.
            requests={"cpu": "50m", "memory": "256Mi", "ephemeral-storage": "100Mi"},
            limits={"memory": "512Mi"},
        ),
    )


def build_mproxy_sidecar(
    image: str,
    envs: list[k8s_client.V1EnvVar],
    volume_mounts: list[k8s_client.V1VolumeMount],
) -> k8s_client.V1Container:
    """GCS tracker sidecar (mitmproxy) intercepting storage.googleapis.com.

    Resource requests keep mproxy from being CPU-starved under main-container
    CPU bursts (TLS handshakes deadlock without guaranteed CPU). Memory
    limits turn OOM into a clean container-level restart instead of
    node-level OOM-killer roulette. Liveness probe catches mitmdump hangs
    (where the process runs but doesn't serve — the in-sidecar self-heal
    loop can only catch actual crashes). The 60s failure window gives the
    self-heal loop room to restart mitmdump between chaos kills without
    the probe itself flagging a false positive.

    No startup probe — the sidecar's self-heal loop handles mitmdump
    readiness internally, and the main container's wait_for_ca script
    gates on the CA-cert file existence (see WAIT_FOR_CA_SCRIPT). A TCP
    startup probe would duplicate that logic and, more importantly,
    kill the container if probe failures accumulated while self-heal was
    already handling things.
    """
    return _build_sidecar_container(
        name="mproxy",
        module="zetta_utils.cloud_management.resource_allocation.k8s.gcs_tracker",
        image=image,
        envs=envs,
        volume_mounts=volume_mounts,
        ports=[k8s_client.V1ContainerPort(container_port=8080, name="proxy")],
        resources=k8s_client.V1ResourceRequirements(
            requests={"cpu": "200m", "memory": "1Gi", "ephemeral-storage": "100Mi"},
            limits={"memory": "2Gi"},
        ),
        liveness_probe=k8s_client.V1Probe(
            tcp_socket=k8s_client.V1TCPSocketAction(port=8080),
            period_seconds=10,
            failure_threshold=6,  # 60s tolerance covers self-heal cycles
            timeout_seconds=3,
        ),
    )
