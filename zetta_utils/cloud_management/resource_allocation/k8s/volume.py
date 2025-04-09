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
CAVE_SECRET_MOUNT_PATH: Final[str] = "/root/.cloudvolume/secrets/"


def get_common_volumes(cave_secret_available: bool = False):
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

    volumes = [dshm, tmp, adc]
    if cave_secret_available:
        cave_secret = k8s_client.V1Volume(
            name="cave-secret",
            secret=k8s_client.V1SecretVolumeSource(secret_name=f"run-{run.RUN_ID}-cave-secret"),
        )
        volumes.append(cave_secret)
    return volumes


def get_common_volume_mounts(cave_secret_available: bool = False):
    mounts = [
        k8s_client.V1VolumeMount(mount_path=SHM_MOUNT_PATH, name="dshm"),
        k8s_client.V1VolumeMount(mount_path=TMP_MOUNT_PATH, name="tmp"),
        k8s_client.V1VolumeMount(
            name="adc",
            mount_path="/etc/secrets",
            read_only=True,
        ),
    ]
    if cave_secret_available:
        mounts.append(
            k8s_client.V1VolumeMount(
                mount_path=CAVE_SECRET_MOUNT_PATH,
                name="cave-secret",
            )
        )
    return mounts
