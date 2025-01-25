"""
Helpers for k8s volumes.
"""

from __future__ import annotations

from typing import Final

from kubernetes import client as k8s_client
from zetta_utils import run

ADC_MOUNT_PATH: Final[str] = "/etc/secrets/adc.json"
SHM_MOUNT_PATH: Final[str] = "/dev/shm"
TMP_MOUNT_PATH: Final[str] = "/tmp"


def get_common_volumes():
    dshm = k8s_client.V1Volume(
        name="dshm", empty_dir=k8s_client.V1EmptyDirVolumeSource(medium="Memory")
    )
    tmp = k8s_client.V1Volume(
        name="tmp", empty_dir=k8s_client.V1EmptyDirVolumeSource(medium="Memory")
    )

    # application_default_credentials
    adc = k8s_client.V1Volume(
        name="adc", secret=k8s_client.V1SecretVolumeSource(secret_name=f"run-{run.RUN_ID}-adc")
    )
    return [dshm, tmp, adc]


def get_common_volume_mounts():
    return [
        k8s_client.V1VolumeMount(mount_path=SHM_MOUNT_PATH, name="dshm"),
        k8s_client.V1VolumeMount(mount_path=TMP_MOUNT_PATH, name="tmp"),
        k8s_client.V1VolumeMount(
            name="adc",
            mount_path="/etc/secrets",
            read_only=True,
        ),
    ]
