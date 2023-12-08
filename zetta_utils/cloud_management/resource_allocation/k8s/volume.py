"""
Helpers for k8s volumes.
"""

from __future__ import annotations

from kubernetes import client as k8s_client  # type: ignore


def get_common_volumes():
    dshm = k8s_client.V1Volume(
        name="dshm", empty_dir=k8s_client.V1EmptyDirVolumeSource(medium="Memory")
    )
    tmp = k8s_client.V1Volume(
        name="tmp", empty_dir=k8s_client.V1EmptyDirVolumeSource(medium="Memory")
    )
    return [dshm, tmp]


def get_common_volume_mounts():
    return [
        k8s_client.V1VolumeMount(mount_path="/dev/shm", name="dshm"),
        k8s_client.V1VolumeMount(mount_path="/tmp", name="tmp"),
    ]
