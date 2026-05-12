"""
Tools to interact with kubernetes clusters.
"""

from .autoscaler import AutoscaleTarget, autoscaling_deployment_ctx_mngr
from .common import (
    ClusterInfo,
    DEFAULT_CLUSTER_INFO,
    DEFAULT_CLUSTER_PROJECT,
    create_dynamic_resource,
    delete_dynamic_resource,
    get_cluster_data,
    parse_cluster_info,
    get_mazepa_worker_command,
)

from .configmap import get_configmap, configmap_ctx_manager
from .cronjob import configure_cronjob
from .deployment import deployment_ctx_mngr, get_deployment, get_mazepa_worker_deployment
from .job import (
    get_job,
    get_job_template,
    get_job_pod,
    get_job_spec,
    job_ctx_manager,
    follow_job_logs,
    wait_for_job_completion,
)
from .pod import (
    ProvisioningModel,
    get_pod_spec,
    get_mazepa_pod_spec,
    get_zone_affinities,
    follow_rank0_logs,
    capture_pod_logs,
    get_pod_postmortem,
)
from .secret import secrets_ctx_mngr, get_secrets_and_mapping
from .service import get_headless_service, get_service, service_ctx_manager
from .statefulset import get_statefulset, statefulset_ctx_manager
from .volume import ADC_MOUNT_PATH, get_common_volumes, get_common_volume_mounts
