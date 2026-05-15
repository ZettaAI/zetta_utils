"""Kubernetes cluster tools — lazily resolved via zetta_utils.common.lazy."""
from typing import TYPE_CHECKING

from zetta_utils.common.lazy import make_lazy_module

_LAZY_REEXPORTS = {
    ".autoscaler": ("AutoscaleTarget", "autoscaling_deployment_ctx_mngr"),
    ".common": (
        "ClusterInfo",
        "DEFAULT_CLUSTER_INFO",
        "DEFAULT_CLUSTER_PROJECT",
        "create_dynamic_resource",
        "delete_dynamic_resource",
        "get_cluster_data",
        "parse_cluster_info",
        "get_mazepa_worker_command",
    ),
    ".configmap": ("get_configmap", "configmap_ctx_manager"),
    ".cronjob": ("configure_cronjob",),
    ".deployment": (
        "deployment_ctx_mngr",
        "get_deployment",
        "get_mazepa_worker_deployment",
    ),
    ".job": (
        "get_job",
        "get_job_template",
        "get_job_pod",
        "get_job_spec",
        "job_ctx_manager",
        "follow_job_logs",
        "wait_for_job_completion",
    ),
    ".pod": (
        "ProvisioningModel",
        "get_pod_spec",
        "get_mazepa_pod_spec",
        "get_zone_affinities",
        "follow_rank0_logs",
        "capture_pod_logs",
        "get_pod_postmortem",
    ),
    ".secret": ("secrets_ctx_mngr", "get_secrets_and_mapping"),
    ".service": ("get_headless_service", "get_service", "service_ctx_manager"),
    ".statefulset": ("get_statefulset", "statefulset_ctx_manager"),
    ".volume": ("ADC_MOUNT_PATH", "get_common_volumes", "get_common_volume_mounts"),
}

__getattr__, __dir__ = make_lazy_module(
    __name__, globals(), reexports_by_module=_LAZY_REEXPORTS
)

if TYPE_CHECKING:
    from .autoscaler import AutoscaleTarget, autoscaling_deployment_ctx_mngr
    from .common import (
        DEFAULT_CLUSTER_INFO,
        DEFAULT_CLUSTER_PROJECT,
        ClusterInfo,
        create_dynamic_resource,
        delete_dynamic_resource,
        get_cluster_data,
        get_mazepa_worker_command,
        parse_cluster_info,
    )
    from .configmap import configmap_ctx_manager, get_configmap
    from .cronjob import configure_cronjob
    from .deployment import (
        deployment_ctx_mngr,
        get_deployment,
        get_mazepa_worker_deployment,
    )
    from .job import (
        follow_job_logs,
        get_job,
        get_job_pod,
        get_job_spec,
        get_job_template,
        job_ctx_manager,
        wait_for_job_completion,
    )
    from .pod import (
        ProvisioningModel,
        capture_pod_logs,
        follow_rank0_logs,
        get_mazepa_pod_spec,
        get_pod_postmortem,
        get_pod_spec,
        get_zone_affinities,
    )
    from .secret import get_secrets_and_mapping, secrets_ctx_mngr
    from .service import get_headless_service, get_service, service_ctx_manager
    from .statefulset import get_statefulset, statefulset_ctx_manager
    from .volume import ADC_MOUNT_PATH, get_common_volume_mounts, get_common_volumes
