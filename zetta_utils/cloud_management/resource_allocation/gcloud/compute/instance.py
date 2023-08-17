"""GCloud Cloud Resource Manager APIs"""

from __future__ import annotations

from typing import MutableSequence, Optional

from google.cloud import compute_v1

from zetta_utils import builder, log

from ..core import wait_for_extended_operation

logger = log.get_logger("zetta_utils")

builder.register("gcloud.AcceleratorConfig")(compute_v1.AcceleratorConfig)


@builder.register("gcloud.create_instance_template")
def create_instance_template(
    template_name: str,
    project: str,
    machine_type: str,
    accelerators: Optional[MutableSequence[compute_v1.AcceleratorConfig]] = None,
    disk_size_gb: int = 64,
    network: str = "default",
    provisioning_model: str = "STANDARD",
    subnetwork: Optional[str] = None,
    source_image: str = "projects/debian-cloud/global/images/family/debian-12",
) -> compute_v1.InstanceTemplate:
    """
    Create an instance template that uses a provided subnet.

    `subnetwork` format - `projects/{project}/regions/{region}/subnetworks/{subnetwork}`
    """

    disk = compute_v1.AttachedDisk()
    initialize_params = compute_v1.AttachedDiskInitializeParams()
    initialize_params.source_image = source_image
    initialize_params.disk_size_gb = disk_size_gb
    disk.initialize_params = initialize_params
    disk.auto_delete = True
    disk.boot = True

    template = compute_v1.InstanceTemplate()
    template.name = template_name
    template.properties = compute_v1.InstanceProperties()
    template.properties.disks = [disk]
    template.properties.machine_type = machine_type
    template.properties.scheduling.provisioning_model = provisioning_model

    if accelerators is not None:
        template.properties.guest_accelerators = accelerators

    network_interface = compute_v1.NetworkInterface()
    network_interface.network = f"projects/{project}/global/networks/{network}"
    if subnetwork is not None:
        network_interface.subnetwork = subnetwork
    template.properties.network_interfaces = [network_interface]

    template_client = compute_v1.InstanceTemplatesClient()
    operation = template_client.insert(project=project, instance_template_resource=template)
    wait_for_extended_operation(operation)
    return template_client.get(project=project, instance_template=template_name)


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
def create_instancegroup_from_template(
    project: str,
    zone: str,
    mig_name: str,
    template_name: str,
    cpu_utilization_target: float = 0.7,
    target_size: int = 1,
    min_replicas: int = 1,
    max_replicas: int = 1,
) -> compute_v1.InstanceGroupManager:
    """
    Creates a Compute Engine VM instance group from an instance template.
    """
    client = compute_v1.InstanceGroupManagersClient()

    request = compute_v1.InsertInstanceGroupManagerRequest()
    request.project = project
    request.zone = zone

    instance_template = f"projects/{project}/global/instanceTemplates/{template_name}"
    request.instance_group_manager_resource.instance_template = instance_template
    request.instance_group_manager_resource.name = mig_name
    request.instance_group_manager_resource.target_size = target_size

    operation = client.insert(request)
    wait_for_extended_operation(operation)
    igmanager = client.get(project=project, zone=zone, instance_group_manager=mig_name)

    assert min_replicas <= max_replicas
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


@builder.register("gcloud.create_mig_autoscaler")
def create_mig_autoscaler(
    project: str,
    zone: str,
    autoscaler_name: str,
    target: str,
    cool_down_period_sec: int = 60,
    cpu_utilization_target: float = 0.7,
    min_replicas: int = 1,
    max_replicas: int = 7,
    mode: str = "ON",
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
