"""
Tools to interact with kubernetes clusters.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

import attrs

from kubernetes import client as k8s_client  # type: ignore
from zetta_utils import builder, log

from .eks import eks_cluster_data
from .gke import gke_cluster_data

logger = log.get_logger("zetta_utils")


@builder.register("mazepa.k8s.ClusterInfo")
@attrs.frozen
class ClusterInfo:
    name: str
    region: Optional[str] = None
    project: Optional[str] = None


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


def get_worker_command(queue_spec: Dict[str, Any]):
    result = (
        """
    zetta -vv -l try run -s '{
        "@type": "mazepa.run_worker"
        exec_queue:
    """
        + json.dumps(queue_spec)
        + """
        max_pull_num: 1
        sleep_sec: 5
    }'
    """
    )
    return result


def _get_provider_cluster_data(info: ClusterInfo) -> Tuple[ClusterAuth, str]:
    if info.project is not None:
        assert info.region is not None, "GKE cluster needs both `project` and `region`."
        logger.info("Cluster provider: GKE/GCP.")

        cluster_data, cert, token = gke_cluster_data(info.name, info.region, info.project)
        endpoint = cluster_data.endpoint
        workload_pool = cluster_data.workload_identity_config.workload_pool
    else:
        logger.info("Cluster provider: EKS/AWS.")

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
