"""GCloud Cloud Resource Manager APIs"""

from enum import Enum

import google.auth
from googleapiclient import discovery

from zetta_utils import log

logger = log.get_logger("zetta_utils")


class Role(Enum):
    WORKLOAD_IDENTITY_USER = "roles/iam.workloadIdentityUser"


def add_role(project_id: str, principal: str, role: Role, member: str) -> None:
    """Adds `member` to `role` binding of the resource."""

    resource = f"projects/{project_id}/serviceAccounts/{principal}"

    policy = get_policy(resource)
    binding = None
    for b in policy["bindings"]:
        if b["role"] == role.value:
            binding = b
            break

    if binding is not None:
        binding["members"].append(member)
    else:
        binding = {"role": role.value, "members": [member]}
        policy["bindings"].append(binding)

    logger.info(f"Adding member {member} to {resource} IAM policy.")
    set_policy(resource, policy)


def remove_role(project_id: str, principal: str, role: Role, member: str) -> None:
    """Removes `member` from the `role` binding."""

    resource = f"projects/{project_id}/serviceAccounts/{principal}"

    policy = get_policy(resource)
    binding = next(b for b in policy["bindings"] if b["role"] == role.value)
    if "members" in binding and member in binding["members"]:
        binding["members"].remove(member)

    logger.info(f"Removing member {member} from {resource} IAM policy.")
    set_policy(resource, policy)


def get_policy(resource: str):
    """Gets IAM policy for the resource."""

    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    service = discovery.build("iam", "v1", credentials=creds)

    request = service.projects().serviceAccounts().getIamPolicy(resource=resource)
    return request.execute()


def set_policy(resource: str, policy):
    """Sets IAM policy for the resource."""

    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    service = discovery.build("iam", "v1", credentials=creds)

    request = (
        service.projects()
        .serviceAccounts()
        .setIamPolicy(resource=resource, body={"policy": policy})
    )
    return request.execute()
