"""GCloud Cloud Resource Manager APIs"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Final, Literal, MutableMapping, MutableSequence, Optional

import yaml
from google.cloud import compute_v1

from zetta_utils import builder, log

from ..core import wait_for_extended_operation

logger = log.get_logger("zetta_utils")

builder.register("gcloud.compute.AcceleratorConfig")(compute_v1.AcceleratorConfig)
builder.register("gcloud.compute.AccessConfig")(compute_v1.AccessConfig)
builder.register("gcloud.compute.ServiceAccount")(compute_v1.ServiceAccount)

COS_INSTALL_GPU_STARTUP_SCRIPT: Final = """#!/bin/bash
sudo cos-extensions install gpu
"""


@builder.register("gcloud.create_instance_template")
def create_instance_template(
    template_name: str,
    project: str,
    bootdisk_size_gb: int,
    machine_type: str,
    source_image: str,
    labels: Optional[MutableMapping[str, str]] = None,
    accelerators: Optional[MutableSequence[compute_v1.AcceleratorConfig]] = None,
    network: str = "default",
    network_accessconfigs: Optional[MutableSequence[compute_v1.AccessConfig]] = None,
    on_host_maintenance: Literal["MIGRATE", "TERMINATE"] = "MIGRATE",
    provisioning_model: Literal["STANDARD", "SPOT"] = "STANDARD",
    service_accounts: Optional[MutableSequence[compute_v1.ServiceAccount]] = None,
    subnetwork: Optional[str] = None,
    worker_image: Optional[str] = None,
    **kwargs,  # pylint: disable=unused-argument
) -> compute_v1.InstanceTemplate:
    """
    Create an instance template that uses a provided subnet.

    `subnetwork` format - `projects/{project}/regions/{region}/subnetworks/{subnetwork}`
    """
    logger.info(f"Creating GCE instance template `{template_name}`")
    if labels is None:
        labels = {}
    labels["created-by"] = os.environ.get("ZETTA_USER", "na")

    if service_accounts is None:
        svc_account = compute_v1.ServiceAccount()
        svc_account.email = "zutils-worker-x0@zetta-research.iam.gserviceaccount.com"
        svc_account.scopes = [
            "https://www.googleapis.com/auth/devstorage.read_write",
            "https://www.googleapis.com/auth/datastore",
            "https://www.googleapis.com/auth/compute",
            "https://www.googleapis.com/auth/cloud-platform",
        ]
        service_accounts = [svc_account]

    bootdisk = compute_v1.AttachedDisk()
    initialize_params = compute_v1.AttachedDiskInitializeParams()
    initialize_params.source_image = source_image
    initialize_params.disk_size_gb = bootdisk_size_gb
    initialize_params.labels = labels
    bootdisk.initialize_params = initialize_params
    bootdisk.auto_delete = True
    bootdisk.boot = True
    disks = [bootdisk]

    template = compute_v1.InstanceTemplate()
    template.name = template_name
    template.properties = compute_v1.InstanceProperties()
    template.properties.labels = labels
    template.properties.disks = disks
    template.properties.machine_type = machine_type
    template.properties.scheduling.provisioning_model = provisioning_model
    template.properties.scheduling.on_host_maintenance = on_host_maintenance
    template.properties.service_accounts = service_accounts

    if accelerators is not None:
        template.properties.guest_accelerators = accelerators
        if "cos" in source_image:
            items = compute_v1.Items()
            if worker_image is None:
                items.key = "startup-script"
                items.value = COS_INSTALL_GPU_STARTUP_SCRIPT
            else:
                items.key = "user-data"
                items.value = _get_worker_cloud_init_config(worker_image)
            template.properties.metadata.items = [items]

    network_interface = compute_v1.NetworkInterface()
    network_interface.network = f"projects/{project}/global/networks/{network}"
    if network_accessconfigs is not None:
        network_interface.access_configs = network_accessconfigs
    if subnetwork is not None:
        network_interface.subnetwork = subnetwork
    template.properties.network_interfaces = [network_interface]

    template_client = compute_v1.InstanceTemplatesClient()
    operation = template_client.insert(project=project, instance_template_resource=template)
    wait_for_extended_operation(operation)
    return template_client.get(project=project, instance_template=template_name)


def delete_instance_template(template_name: str, project: str):
    logger.info(f"Deleting GCE instance template `{template_name}`")
    client = compute_v1.InstanceTemplatesClient()
    request = compute_v1.DeleteInstanceTemplateRequest()
    # template_path = f"projects/{project}/global/instanceTemplates/{template_name}"
    request.instance_template = template_name
    request.project = project
    operation = client.delete(request)
    wait_for_extended_operation(operation)


@builder.register("gcloud.instance_template_ctx_mngr")
@contextmanager
def instance_template_ctx_mngr(**kwargs):
    create_instance_template(**kwargs)
    try:
        yield
    finally:
        delete_instance_template(kwargs["template_name"], kwargs["project"])


@builder.register("gcloud.create_instance_from_template")
def create_instance_from_template(
    project: str,
    zone: str,
    instance_name: str,
    template_name: str,
) -> compute_v1.Instance:
    """
    Creates a Compute Engine VM instance from an instance template.
    """
    client = compute_v1.InstancesClient()

    request = compute_v1.InsertInstanceRequest()
    request.project = project
    request.zone = zone

    instance_template = f"projects/{project}/global/instanceTemplates/{template_name}"
    request.source_instance_template = instance_template
    request.instance_resource.name = instance_name

    operation = client.insert(request)
    wait_for_extended_operation(operation)
    return client.get(project=project, zone=zone, instance=instance_name)


@builder.register("gcloud.create_mig_from_template")
def create_mig_from_template(
    project: str,
    zone: str,
    mig_name: str,
    template_name: str,
    cpu_utilization_target: float = 0.7,
    target_size: int = 0,
    min_replicas: int = 1,
    max_replicas: int = 1,
    **kwargs,  # pylint: disable=unused-argument
) -> compute_v1.InstanceGroupManager:
    """
    Creates a Compute Engine VM instance group from an instance template.
    """
    logger.info(f"Creating GCE Managed Instance Group `{mig_name}`")
    assert min_replicas <= max_replicas

    client = compute_v1.InstanceGroupManagersClient()
    request = compute_v1.InsertInstanceGroupManagerRequest()
    request.project = project
    request.zone = zone

    template_path = f"projects/{project}/global/instanceTemplates/{template_name}"
    request.instance_group_manager_resource.instance_template = template_path
    request.instance_group_manager_resource.name = mig_name
    request.instance_group_manager_resource.target_size = target_size

    operation = client.insert(request)
    wait_for_extended_operation(operation)
    igmanager = client.get(project=project, zone=zone, instance_group_manager=mig_name)

    if min_replicas < max_replicas:
        create_mig_autoscaler(
            project=project,
            zone=zone,
            autoscaler_name=f"{mig_name}-autoscaler",
            target=igmanager.self_link,
            cpu_utilization_target=cpu_utilization_target,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
        )
    return igmanager


def delete_mig(mig_name: str, project: str, zone: str):
    logger.info(f"Deleting GCE Managed Instance Group `{mig_name}`")
    client = compute_v1.InstanceGroupManagersClient()
    request = compute_v1.DeleteInstanceGroupManagerRequest()
    request.instance_group_manager = mig_name
    request.project = project
    request.zone = zone
    operation = client.delete(request)
    wait_for_extended_operation(operation)


@builder.register("gcloud.mig_ctx_mngr")
@contextmanager
def mig_ctx_mngr(**kwargs):
    create_mig_from_template(**kwargs)
    try:
        yield
    finally:
        delete_mig(kwargs["mig_name"], kwargs["project"], kwargs["zone"])


@builder.register("gcloud.create_mig_autoscaler")
def create_mig_autoscaler(
    project: str,
    zone: str,
    autoscaler_name: str,
    target: str,
    cool_down_period_sec: int = 60,
    cpu_utilization_target: float = 0.7,
    min_replicas: int = 1,
    max_replicas: int = 1,
    mode: Literal["ON", "OFF", "ONLY_SCALE_OUT"] = "ON",
) -> compute_v1.Autoscaler:
    """
    Creates a Compute Engine Autoscaler for a Managed Instance Group (MIG).

    `target` format - `projects/{project}/zones/{zone}/instanceGroups/{mig_name}`
    """
    client = compute_v1.AutoscalersClient()

    request = compute_v1.InsertAutoscalerRequest()
    request.project = project
    request.zone = zone
    request.autoscaler_resource.target = target
    request.autoscaler_resource.name = autoscaler_name

    autoscaling_policy = request.autoscaler_resource.autoscaling_policy
    autoscaling_policy.cool_down_period_sec = cool_down_period_sec
    autoscaling_policy.cpu_utilization.utilization_target = cpu_utilization_target
    autoscaling_policy.min_num_replicas = min_replicas
    autoscaling_policy.max_num_replicas = max_replicas
    autoscaling_policy.mode = mode

    operation = client.insert(request)
    wait_for_extended_operation(operation)
    return client.get(project=project, zone=zone, autoscaler=autoscaler_name)


def get_instance(
    project: str,
    zone: str,
    instance_name: str,
) -> compute_v1.Instance:
    """Gets an existing VM instance."""
    client = compute_v1.InstancesClient()
    request = compute_v1.GetInstanceRequest()
    request.project = project
    request.zone = zone
    return client.get(project=project, zone=zone, instance=instance_name)


def _get_worker_cloud_init_config(image: str):
    user = {"user": os.environ["ZETTA_USER"], "uid": 2000}
    gpu_service_file = {
        "path": "/etc/systemd/system/install-gpu.service",
        "content": f"""[Unit]
Description=Install GPU drivers
Wants=gcr-online.target docker.socket
After=gcr-online.target docker.socket

[Service]
User={os.environ['ZETTA_USER']}
Type=oneshot
ExecStart=cos-extensions install gpu
StandardOutput=journal+console
StandardError=journal+console
""",
    }

    mounts_ = [
        "--volume /var/lib/nvidia:/usr/local/nvidia",
        "--device /dev/nvidia-uvm:/dev/nvidia-uvm",
        "--device /dev/nvidiactl:/dev/nvidiactl",
        "--device /dev/nvidia0:/dev/nvidia0",
    ]
    mounts = " ".join(mounts_)

    envs_ = [
        "-e MY_ROLE=worker",
        "-e MASTER_ADDR=master",
        "-e MASTER_PORT=29400",
        "-e LOGLEVEL=DEBUG",
        "-e NCCL_DEBUG=INFO",
        "-e NCCL_SOCKET_IFNAME=eth0",
        f"-e ZETTA_USER={os.environ['ZETTA_USER']}",
        f"-e ZETTA_PROJECT={os.environ['ZETTA_PROJECT']}",
        f"-e WANDB_API_KEY={os.environ['WANDB_API_KEY']}",
    ]
    envs = " ".join(envs_)
    args = f"--network=host --add-host master:{os.environ['NODE_IP']} {envs} {mounts}"

    workers_service_file = {
        "path": "/etc/systemd/system/workers.service",
        "content": f"""[Unit]
Description=Run a workers GPU application container
Requires=install-gpu.service
After=install-gpu.service

[Service]
User={os.environ['ZETTA_USER']}
Type=oneshot
RemainAfterExit=true
ExecStart=docker run --name=workers {args} {image} zetta run -s '{os.environ["ZETTA_RUN_SPEC"]}'
StandardOutput=journal+console
StandardError=journal+console
""",
    }

    cloud_init_config = {
        "users": [user],
        "write_files": [gpu_service_file, workers_service_file],
        "runcmd": [
            "systemctl daemon-reload",
            "systemctl start install-gpu.service",
            f"su {os.environ['ZETTA_USER']} -c 'docker-credential-gcr configure-docker'",
            "systemctl start workers.service",
        ],
    }

    def _multiline_presenter(dumper, data):
        if data.count("\n") > 0:
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
        return dumper.represent_scalar("tag:yaml.org,2002:str", data)

    yaml.add_representer(str, _multiline_presenter)
    return f"#cloud-config\n\n{yaml.dump(cloud_init_config, default_flow_style=False)}"
