"""
Fetch EKS Cluster info.
"""
import base64
from tempfile import NamedTemporaryFile
from typing import Any, Tuple

import boto3
from awscli.customizations.eks.get_token import STSClientFactory, TokenGenerator
from botocore import session


def get_eks_token(cluster_name: str) -> str:
    work_session = session.get_session()
    client_factory = STSClientFactory(work_session)
    sts_client = client_factory.get_sts_client()
    return TokenGenerator(sts_client).get_token(cluster_name)


def eks_cluster_data(name: str) -> Tuple[Any, str, str]:
    bclient = boto3.client("eks")
    cluster_response = bclient.describe_cluster(name=name)
    cluster_data = cluster_response["cluster"]
    cert_auth = cluster_data["certificateAuthority"]

    with NamedTemporaryFile(delete=False) as ca_cert:
        ca_cert.write(base64.b64decode(cert_auth["data"]))
        cert_name = ca_cert.name

    token = get_eks_token(name)
    return cluster_data, cert_name, token
