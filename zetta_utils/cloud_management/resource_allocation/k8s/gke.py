"""
Fetch GKE Cluster info.
"""
import base64
from tempfile import NamedTemporaryFile
from typing import Tuple

from google import auth as google_auth
from google.cloud.container_v1 import Cluster, ClusterManagerClient


def gke_cluster_data(name: str, region: str, project: str) -> Tuple[Cluster, str, str]:
    cluster_name = f"projects/{project}/locations/{region}/clusters/{name}"

    # see https://github.com/googleapis/python-container/issues/6
    container_client = ClusterManagerClient()
    cluster_data = container_client.get_cluster(name=cluster_name)

    with NamedTemporaryFile(delete=False) as ca_cert:
        ca_cert.write(base64.b64decode(cluster_data.master_auth.cluster_ca_certificate))
        cert_name = ca_cert.name

    creds, _ = google_auth.default()
    creds.refresh(google_auth.transport.requests.Request())
    return cluster_data, cert_name, creds.token
