"""Manage GCP VM instances."""

from datetime import datetime

from googleapiclient import discovery


def get_node_info(project_id: str, zone: str, instance_name: str):
    """
    Get information about a VM instance in a given zone.
    Returns machine type, resources, creation time and provisioning model.
    """
    service = discovery.build("compute", "v1")

    request = service.instances().get(project=project_id, zone=zone, instance=instance_name)
    response = request.execute()

    machine_type_url = response.get("machineType", "")
    machine_type = machine_type_url.split("/")[-1]

    machine_type_req = service.machineTypes().get(
        project=project_id, zone=zone, machineType=machine_type
    )
    machine_type_resp = machine_type_req.execute()
    cpu_count = machine_type_resp.get("guestCpus")
    memory_mb = machine_type_resp.get("memoryMb")
    memory_gib = memory_mb / 1024 if memory_mb else None

    accelerators = response.get("guestAccelerators", [])
    gpus = {}
    for acc in accelerators:
        gpus[acc.get("acceleratorType").split("/")[-1]] = acc.get("acceleratorCount")

    creation_time = response.get("creationTimestamp")
    dt = datetime.fromisoformat(creation_time)
    creation_time_epoch = int(dt.timestamp())

    scheduling = response.get("scheduling", {})
    provisioning_model = "STANDARD"
    if response.get("scheduling", {}).get("preemptible", False):
        provisioning_model = "PREEMPTIBLE"
    elif scheduling.get("provisioningModel"):
        provisioning_model = scheduling.get("provisioningModel")

    return {
        "machine_type": machine_type,
        "cpu_count": cpu_count,
        "memory_gib": memory_gib,
        "gpus": gpus,
        "creation_time": creation_time_epoch,
        "provisioning_model": provisioning_model,
    }
