"""
Helpers for k8s job.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from kubernetes import client as k8s_client  # type: ignore
from zetta_utils import log

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

    container = k8s_client.V1Container(
        command=command,
        args=command_args,
        env=envs,
        name=name,
        image=image,
        image_pull_policy="IfNotPresent",
        resources=k8s_client.V1ResourceRequirements(
            requests=resources,
            limits=resources,
        ),
        termination_message_path="/dev/termination-log",
        termination_message_policy="File",
        volume_mounts=[],
    )

    schedule_toleration = k8s_client.V1Toleration(
        key="worker-pool", operator="Equal", value="true", effect="NoSchedule"
    )

    pod_spec = k8s_client.V1PodSpec(
        containers=[container],
        dns_policy="Default",
        restart_policy="OnFailure",
        scheduler_name="default-scheduler",
        security_context={},
        termination_grace_period_seconds=30,
        tolerations=[schedule_toleration],
    )

    common_meta = k8s_client.V1ObjectMeta(name=name, namespace=namespace, labels=labels)

    pod_template = k8s_client.V1PodTemplateSpec(metadata=common_meta, spec=pod_spec)
    job_spec = k8s_client.V1JobSpec(template=pod_template)

    return k8s_client.V1JobTemplateSpec(metadata=common_meta, spec=job_spec)
