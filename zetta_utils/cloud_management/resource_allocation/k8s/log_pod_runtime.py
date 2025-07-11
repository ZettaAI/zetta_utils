import os
import time

import requests

from kubernetes import client as k8s_client
from kubernetes import config  # type: ignore
from zetta_utils.cloud_management.resource_allocation import gcloud
from zetta_utils.run.db import NODE_DB


def get_project_id():
    metadata_url = "http://metadata.google.internal/computeMetadata/v1/project/project-id"
    headers = {"Metadata-Flavor": "Google"}
    response = requests.get(metadata_url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.text


def log_pod_runtime():
    config.load_incluster_config()
    api = k8s_client.CoreV1Api()
    pod_name = os.environ["POD_NAME"]
    run_id = os.environ["RUN_ID"]
    pod = api.read_namespaced_pod(name=pod_name, namespace="default")
    node_name = pod.spec.node_name

    if "default" in node_name or "system" in node_name:
        return

    node = api.read_node(node_name)
    node_zone = node.metadata.labels["topology.kubernetes.io/zone"]
    project_id = get_project_id()
    node_info = gcloud.get_node_info(
        project_id=project_id, zone=node_zone, instance_name=node_name
    )
    node_info["+run_id"] = [run_id]
    node_info[run_id] = time.time()
    NODE_DB[node_name] = node_info
