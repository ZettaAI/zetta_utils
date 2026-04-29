"""
Fetch GKE Cluster info and resize node pools.
"""
import base64
from tempfile import NamedTemporaryFile
from typing import Tuple

from google import auth as google_auth
from google.cloud.container_v1 import Cluster, ClusterManagerClient, NodePool
from google.cloud.container_v1.types import Operation, SetNodePoolSizeRequest

_cluster_manager: ClusterManagerClient | None = None


def _get_cluster_manager() -> ClusterManagerClient:
    """Process-wide singleton ``ClusterManagerClient``.

    Constructing the client opens a gRPC channel and runs ADC auth, so
    callers on the autoscaler / nudger hot paths share one instance.
    """
    global _cluster_manager  # pylint: disable=global-statement
    if _cluster_manager is None:
        _cluster_manager = ClusterManagerClient()
    return _cluster_manager


def gke_cluster_data(name: str, region: str, project: str) -> Tuple[Cluster, str, str]:
    cluster_name = f"projects/{project}/locations/{region}/clusters/{name}"

    # see https://github.com/googleapis/python-container/issues/6
    cluster_data = _get_cluster_manager().get_cluster(name=cluster_name)

    with NamedTemporaryFile(delete=False) as ca_cert:
        ca_cert.write(base64.b64decode(cluster_data.master_auth.cluster_ca_certificate))
        cert_name = ca_cert.name

    # creds, _ = google_auth.default()
    creds, _ = google_auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(google_auth.transport.requests.Request())
    assert creds.token is not None
    return cluster_data, cert_name, creds.token


def list_node_pools(project: str, location: str, cluster: str) -> list[NodePool]:
    """List all node pools of a GKE cluster.

    Each :class:`NodePool` exposes ``initial_node_count`` (the current
    per-zone target, despite the proto name), ``locations`` (zones),
    ``instance_group_urls`` (underlying GCE MIGs), ``autoscaling``
    (min/max node counts), and ``max_pods_constraint.max_pods_per_node``.
    """
    parent = f"projects/{project}/locations/{location}/clusters/{cluster}"
    return list(_get_cluster_manager().list_node_pools(parent=parent).node_pools)


def resize_node_pool(
    project: str,
    location: str,
    cluster: str,
    pool_name: str,
    node_count: int,
) -> Operation:
    """Set a node pool's per-zone target node count.

    Wraps ``ClusterManagerClient.set_node_pool_size``. Returns the
    long-running operation immediately; callers are expected to
    fire-and-forget. Idempotent — setting to the current size is a no-op.
    For regional pools, ``node_count`` is the per-zone count, not the
    total across zones.
    """
    name = f"projects/{project}/locations/{location}" f"/clusters/{cluster}/nodePools/{pool_name}"
    request = SetNodePoolSizeRequest(name=name, node_count=node_count)
    return _get_cluster_manager().set_node_pool_size(request=request)
