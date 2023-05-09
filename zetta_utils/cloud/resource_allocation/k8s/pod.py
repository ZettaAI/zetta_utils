"""
Helpers for k8s pod.
"""

from typing import Dict, List, Optional

from kubernetes import client as k8s_client  # type: ignore
from zetta_utils import log

from .secret import get_worker_env_vars

logger = log.get_logger("zetta_utils")


def get_pod_spec(
    name: str,
    image: str,
    command: List[str],
    command_args: List[str],
    resources: Dict[str, int | float | str],
    dns_policy: Optional[str] = "Default",
    image_pull_policy: Optional[str] = "IfNotPresent",
    envs: Optional[List[k8s_client.V1EnvVar]] = None,
    env_secret_mapping: Optional[Dict[str, str]] = None,
    restart_policy: Optional[str] = "Always",
    tolerations: Optional[List[k8s_client.V1Toleration]] = None,
) -> k8s_client.V1PodSpec:
    if envs is None:
        envs = []

    if tolerations is None:
        tolerations = []

    volume_mounts = [
        k8s_client.V1VolumeMount(mount_path="/dev/shm", name="dshm"),
        k8s_client.V1VolumeMount(mount_path="/tmp", name="tmp"),
    ]

    container = k8s_client.V1Container(
        command=command,
        args=command_args,
        env=get_worker_env_vars(env_secret_mapping),
        name=name,
        image=image,
        image_pull_policy=image_pull_policy,
        resources=k8s_client.V1ResourceRequirements(
            requests=resources,
            limits=resources,
        ),
        termination_message_path="/dev/termination-log",
        termination_message_policy="File",
        volume_mounts=volume_mounts,
    )

    dshm = k8s_client.V1Volume(
        name="dshm", empty_dir=k8s_client.V1EmptyDirVolumeSource(medium="Memory")
    )
    tmp = k8s_client.V1Volume(
        name="tmp", empty_dir=k8s_client.V1EmptyDirVolumeSource(medium="Memory")
    )

    return k8s_client.V1PodSpec(
        containers=[container],
        dns_policy=dns_policy,
        restart_policy=restart_policy,
        scheduler_name="default-scheduler",
        security_context={},
        termination_grace_period_seconds=30,
        tolerations=tolerations,
        volumes=[dshm, tmp],
    )
