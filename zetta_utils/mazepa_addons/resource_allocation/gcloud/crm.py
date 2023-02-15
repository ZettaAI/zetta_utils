"""GCloud Cloud Resource Manager APIs"""

from enum import Enum
from typing import Optional

import google.auth
import googleapiclient.discovery


class Role(Enum):
    WORKLOAD_IDENTITY_USER = "roles/iam.workloadIdentityUser"


def add_role(project_id: str, role: Role, member: str, principal: Optional[str] = None) -> None:
    """Adds `member` to `role` binding of the resource."""

    resource = project_id
    if principal is None:
        resource = f"projects/-/serviceAccounts/{principal}"

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

    set_policy(project_id, policy)


def remove_role(project_id: str, role: Role, member: str, principal: Optional[str] = None) -> None:
    """Removes `member` from the `role` binding."""

    resource = project_id
    if principal is None:
        resource = f"projects/-/serviceAccounts/{principal}"

    policy = get_policy(resource)
    binding = next(b for b in policy["bindings"] if b["role"] == role.value)
    if "members" in binding and member in binding["members"]:
        binding["members"].remove(member)

    set_policy(project_id, policy)


def get_policy(resource: str, version=3):
    """Gets IAM policy for the resource."""

    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    crm = googleapiclient.discovery.build("cloudresourcemanager", "v1", credentials=creds)

    policy = (
        crm.projects()
        .getIamPolicy(
            resource=resource,
            body={"options": {"requestedPolicyVersion": version}},
        )
        .execute()
    )
    return policy


def set_policy(resource: str, policy):
    """Sets IAM policy for the resource."""

    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    crm = googleapiclient.discovery.build("cloudresourcemanager", "v1", credentials=creds)
    policy = crm.projects().setIamPolicy(resource=resource, body={"policy": policy}).execute()
    return policy
