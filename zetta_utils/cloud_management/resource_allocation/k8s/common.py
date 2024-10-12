"""
Tools to interact with kubernetes clusters.
"""

from __future__ import annotations

import json
from typing import Any, Final, Optional, Tuple

import attrs
from kubernetes.dynamic import DynamicClient

from kubernetes import client as k8s_client  # type: ignore
from zetta_utils import builder, log, run
from zetta_utils.mazepa import SemaphoreType

from .eks import eks_cluster_data
from .gke import gke_cluster_data

logger = log.get_logger("zetta_utils")


DEFAULT_CLUSTER_NAME: Final = "zutils-x3"
DEFAULT_CLUSTER_REGION: Final = "us-east1"
DEFAULT_CLUSTER_PROJECT: Final = "zetta-research"


@builder.register("mazepa.k8s.ClusterInfo")
@attrs.frozen
class ClusterInfo:
    name: str
    region: Optional[str] = None
    project: Optional[str] = None


DEFAULT_CLUSTER_INFO: Final = ClusterInfo(
    name=DEFAULT_CLUSTER_NAME,
    region=DEFAULT_CLUSTER_REGION,
    project=DEFAULT_CLUSTER_PROJECT,
)


@attrs.frozen
class ClusterAuth:
    cert_name: str
    endpoint: str
    token: str


def _get_init_container_command() -> str:
    command = """
    curl -sS -H 'Metadata-Flavor: Google' 'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token' \
        --retry 30 \
        --retry-connrefused \
        --retry-max-time 60 \
        --connect-timeout 3 \
        --fail \
        --retry-all-errors > /dev/null && \
        exit 0 || echo 'Retry limit exceeded. Check if the gke-metadata-server is healthy.' >&2; exit 1
    """
    return command


def get_mazepa_worker_command(
    task_queue_spec: dict[str, Any],
    outcome_queue_spec: dict[str, Any],
    num_procs: int = 1,
    semaphores_spec: dict[SemaphoreType, int] | None = None,
):
    if num_procs == 1 and semaphores_spec is None:
        command = "mazepa.run_worker"
        num_procs_line = ""
        semaphores_line = ""
    else:
        command = "mazepa.run_worker_manager"
        num_procs_line = f"num_procs: {num_procs}\n"
        semaphores_line = f"semaphores_spec: {json.dumps(semaphores_spec)}\n"

    result = f"zetta -vv -l try run -r {run.RUN_ID} --no-main-run-process -p -s '{{"
    result += (
        f'"@type": "{command}"\n'
        + f"task_queue: {json.dumps(task_queue_spec)}\n"
        + f"outcome_queue: {json.dumps(outcome_queue_spec)}\n"
        + num_procs_line
        + semaphores_line
        + """
        sleep_sec: 5
    }'
    """
    )
    return result


def _get_provider_cluster_data(info: ClusterInfo) -> Tuple[ClusterAuth, str]:
    if info.project is not None:
        assert info.region is not None, "GKE cluster needs both `project` and `region`."
        logger.debug("Cluster provider: GKE/GCP.")

        cluster_data, cert, token = gke_cluster_data(info.name, info.region, info.project)
        endpoint = cluster_data.endpoint
        workload_pool = cluster_data.workload_identity_config.workload_pool
    else:
        logger.debug("Cluster provider: EKS/AWS.")

        cluster_data, cert, token = eks_cluster_data(info.name)
        endpoint = cluster_data["endpoint"]
        workload_pool = cluster_data["workload_identity_config"]["workload_pool"]

    return ClusterAuth(cert, endpoint, token), workload_pool


def get_cluster_data(info: ClusterInfo) -> Tuple[k8s_client.Configuration, str]:
    cluster_auth, workload_pool = _get_provider_cluster_data(info)

    logger.debug(f"Cluster endpoint: {cluster_auth.endpoint}")
    configuration = k8s_client.Configuration()
    configuration.host = f"https://{cluster_auth.endpoint}"
    configuration.ssl_ca_cert = cluster_auth.cert_name
    configuration.api_key_prefix["authorization"] = "Bearer"
    configuration.api_key["authorization"] = cluster_auth.token
    return configuration, workload_pool


def parse_cluster_info(
    cluster_name: Optional[str] = None,
    cluster_region: Optional[str] = None,
    cluster_project: Optional[str] = None,
) -> ClusterInfo:
    if cluster_name is None:
        logger.info(f"Using default cluster: {DEFAULT_CLUSTER_INFO}")
        cluster_info = DEFAULT_CLUSTER_INFO
        if cluster_region is not None or cluster_project is not None:
            raise ValueError(
                "Both `cluster_region` and `cluster_project` must be `None` "
                "when `cluster_name` is `None`"
            )
    else:
        if cluster_region is None or cluster_project is None:
            raise ValueError(
                "Both `cluster_region` and `cluster_project` must be provided "
                "when `cluster_name` is specified."
            )
        cluster_info = ClusterInfo(
            name=cluster_name,
            region=cluster_region,
            project=cluster_project,
        )
    return cluster_info


def create_dynamic_resource(
    name: str,
    configuration: k8s_client.Configuration,
    api_version: str,
    kind: str,
    manifest: dict,
    namespace: str | None = "default",
):
    api_client = k8s_client.api_client
    client = DynamicClient(api_client.ApiClient(configuration=configuration))
    dynamic_api = client.resources.get(api_version=api_version, kind=kind)
    logger.info(f"Creating dynamic k8s resource `{name}`")
    dynamic_api.create(body=manifest, namespace=namespace)


def delete_dynamic_resource(
    name: str,
    configuration: k8s_client.Configuration,
    api_version: str,
    kind: str,
    namespace: str | None = "default",
):
    api_client = k8s_client.api_client
    client = DynamicClient(api_client.ApiClient(configuration=configuration))
    dynamic_api = client.resources.get(api_version=api_version, kind=kind)
    logger.info(f"Deleting dynamic k8s resource `{name}`")
    dynamic_api.delete(name=name, body={}, namespace=namespace)
