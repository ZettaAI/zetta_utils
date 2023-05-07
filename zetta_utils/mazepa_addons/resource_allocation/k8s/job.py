"""
Helpers for k8s job.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from kubernetes import client as k8s_client  # type: ignore
from zetta_utils import log

from .pod import get_pod_spec

logger = log.get_logger("zetta_utils")


def get_job_template(
    name: str,
    namespace: str,
    image: str,
    command: List[str],
    command_args: List[str],
    envs: List[k8s_client.V1EnvVar],
    resources: Dict[str, int | float | str],
    labels: Optional[Dict[str, str]] = None,
) -> k8s_client.V1Job:

    pod_spec = get_pod_spec(
        name=name,
        image=image,
        command=command,
        command_args=command_args,
        envs=envs,
        resources=resources,
    )

    common_meta = k8s_client.V1ObjectMeta(name=name, namespace=namespace, labels=labels)

    pod_template = k8s_client.V1PodTemplateSpec(metadata=common_meta, spec=pod_spec)
    job_spec = k8s_client.V1JobSpec(template=pod_template)

    return k8s_client.V1JobTemplateSpec(metadata=common_meta, spec=job_spec)
