"""
Helpers for k8s secrets.
"""

import base64
import json
import os
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Tuple

from cloudvolume.secrets import cave_credentials

from kubernetes import client as k8s_client
from zetta_utils import log
from zetta_utils.run.resource import (
    Resource,
    ResourceTypes,
    deregister_resource,
    register_resource,
)

from .common import ClusterInfo, get_cluster_data

logger = log.get_logger("zetta_utils")


def get_worker_env_vars(env_secret_mapping: Optional[Dict[str, str]] = None) -> list:
    if env_secret_mapping is None:
        env_secret_mapping = {}
    name_path_map = {
        "MY_NODE_NAME": "spec.nodeName",
        "MY_POD_NAME": "metadata.name",
        "MY_POD_NAMESPACE": "metadata.namespace",
        "MY_POD_IP": "status.podIP",
        "MY_POD_SERVICE_ACCOUNT": "spec.serviceAccountName",
    }
    envs = [
        k8s_client.V1EnvVar(
            name=name,
            value_from=k8s_client.V1EnvVarSource(
                field_ref=k8s_client.V1ObjectFieldSelector(field_path=path)
            ),
        )
        for name, path in name_path_map.items()
    ]

    for k, v in env_secret_mapping.items():
        env_var = k8s_client.V1EnvVar(
            name=k,
            value_from=k8s_client.V1EnvVarSource(
                secret_key_ref=k8s_client.V1SecretKeySelector(key="value", name=v, optional=False)
            ),
        )
        envs.append(env_var)
    return envs


def _get_user_adc() -> Optional[str]:
    """
    Reads credentials file created by `gcloud auth application-default login`.
    Returns base64 encoded credentials or None if not found.
    """
    # Try environment variable path first, then default location
    credential_paths = [
        os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
        os.path.join(
            os.path.expanduser("~"), ".config/gcloud/application_default_credentials.json"
        ),
    ]

    # Find first existing path
    file_path = next((path for path in credential_paths if path and os.path.exists(path)), None)

    if not file_path:
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return base64.b64encode(f.read().encode()).decode()
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def get_secrets_and_mapping(
    run_id: str, share_envs: Iterable[str] = ()
) -> Tuple[List[k8s_client.V1Secret], Dict[str, str], bool, bool]:
    env_secret_mapping: Dict[str, str] = {}
    secrets_kv: Dict[str, str] = {}
    adc_available: bool = False
    cave_secret_available: bool = False

    combined_secret_data = {}
    for env_k in share_envs:
        if not env_k.isupper() or not env_k.replace("_", "").isalpha():
            raise ValueError(
                "Only able to share environment variables with "
                f"only upper letters and underscores. Got: `{env_k}`"
            )
        env_v = os.environ.get(env_k, None)
        if env_v is None:
            raise ValueError(
                f"Please set `{env_k}` environment variable in order to create a deployment."
            )
        combined_secret_data[env_k] = env_v
        secret_name = f"run-{run_id}-{env_k}".lower().replace("_", "-")
        env_secret_mapping[env_k] = secret_name
        secrets_kv[secret_name] = env_v

    # this is necessary for keda sqs trigger
    combined_secret = k8s_client.V1Secret(
        metadata=k8s_client.V1ObjectMeta(name=f"run-{run_id}-secret-combined"),
        string_data=combined_secret_data,
    )
    secrets = [combined_secret]
    for k, v in secrets_kv.items():
        secret = k8s_client.V1Secret(
            metadata=k8s_client.V1ObjectMeta(name=k),
            string_data={"value": v},
        )
        secrets.append(secret)

    adc_content = _get_user_adc()
    adc_available = not adc_content is None
    adc_creds = k8s_client.V1Secret(
        metadata=k8s_client.V1ObjectMeta(name=f"run-{run_id}-adc"),
        data={"adc.json": adc_content},
    )
    cave_secret_content = cave_credentials()
    cave_secret_available = cave_secret_content != {}
    cave_secret_base64 = base64.b64encode(json.dumps(cave_secret_content).encode("utf-8")).decode(
        "utf-8"
    )
    cave_secret_creds = k8s_client.V1Secret(
        metadata=k8s_client.V1ObjectMeta(name=f"run-{run_id}-cave-secret"),
        data={"cave-secret.json": cave_secret_base64},
    )

    secrets.append(adc_creds)
    secrets.append(cave_secret_creds)
    return secrets, env_secret_mapping, adc_available, cave_secret_available


@contextmanager
def secrets_ctx_mngr(
    run_id: str,
    secrets: List[k8s_client.V1Secret],
    cluster_info: ClusterInfo,
    namespace: Optional[str] = "default",
):
    configuration, _ = get_cluster_data(cluster_info)
    k8s_client.Configuration.set_default(configuration)
    k8s_core_v1_api = k8s_client.CoreV1Api()
    secrets_resource_ids = []
    for secret in secrets:
        logger.info(f"Creating k8s secret `{secret.metadata.name}`")
        k8s_core_v1_api.create_namespaced_secret(namespace=namespace, body=secret)
        _id = register_resource(
            Resource(
                run_id,
                ResourceTypes.K8S_SECRET.value,
                secret.metadata.name,
            )
        )
        secrets_resource_ids.append(_id)

    try:
        yield
    finally:
        # new configuration to refresh expired tokens (long running executions)
        configuration, _ = get_cluster_data(cluster_info)
        k8s_client.Configuration.set_default(configuration)

        # need to create a new client for the above to take effect
        k8s_core_v1_api = k8s_client.CoreV1Api()
        for secret, _id in zip(secrets, secrets_resource_ids):
            logger.info(f"Deleting k8s secret `{secret.metadata.name}`")
            k8s_core_v1_api.delete_namespaced_secret(
                name=secret.metadata.name, namespace=namespace
            )
            deregister_resource(_id)
