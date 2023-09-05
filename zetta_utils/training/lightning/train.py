from __future__ import annotations

import json
import os
from contextlib import ExitStack
from typing import Any, Dict, Final, List, Optional

import pytorch_lightning as pl
import torch
import typeguard
from pytorch_lightning.strategies import ddp
from pytorch_lightning.utilities.cloud_io import get_filesystem
from torch.distributed.launcher import api as torch_launcher_api

from kubernetes import client as k8s_client  # type: ignore
from zetta_utils import builder, load_all_modules, log, mazepa, parsing
from zetta_utils.cloud_management import execution_tracker, resource_allocation

logger = log.get_logger("zetta_utils")

builder.register("pl.Trainer")(pl.Trainer)
builder.register("pl.callbacks.ModelCheckpoint")(pl.callbacks.ModelCheckpoint)
builder.register("pl.DDPStrategy")(ddp.DDPStrategy)

REQUIRED_ENV_VARS: Final = [
    "ZETTA_USER",
    "ZETTA_PROJECT",
    "WANDB_API_KEY",
]


@builder.register("lightning_train")
@typeguard.typechecked
def lightning_train(
    regime: pl.LightningModule,
    trainer: pl.Trainer,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader | None = None,
    full_state_ckpt_path: str = "last",
):
    """
    Perform neural net trainig with Zetta's PytorchLightning integration.

    :param regime: Training regime. Defines behavior on training, vallidation steps
        and epochs. Includes the model being trained as an instance variable.
    :param trainer: Pytorch Lightning Trainer object responsible for handling
        traing loop details that are common for all regimes, such as checkpointing
        behavior, logging behavior, etc. For Zetta training configuration, use
        ``zetta_utils.training.lightning.trainers.build_default_trainer``.
    :param train_dataloader: Training dataloader.
    :param val_dataloader: Validation dataloader.
    :param full_state_ckpt_path: Path to the training checkpoint to resume from.
        Must be a full training state checkpoint created by PytorchLightning rather
        than a model checkpoint. If ``full_state_ckpt_path=="last"``, the latest
        checkpoint for the given experiment will be identified and loaded.
    """
    logger.info("Starting training...")
    if "CURRENT_BUILD_SPEC" in os.environ:
        if hasattr(trainer, "log_config"):
            trainer.log_config(json.loads(os.environ["CURRENT_BUILD_SPEC"]))
        else:
            logger.warning("Incompatible custom trainer used: Unable to save configuration.")
    else:
        logger.warning("Invoked without builder: Unable to save configuration.")

    if full_state_ckpt_path == "last":
        if get_filesystem(trainer.ckpt_path).exists(trainer.ckpt_path):  # type: ignore
            ckpt_path = trainer.ckpt_path
        else:
            ckpt_path = None
    trainer.fit(
        model=regime,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=ckpt_path,
    )


def _parse_spec_and_train():
    load_all_modules()
    regime = builder.build(spec=json.loads(os.environ["ZETTA_RUN_SPEC"])["regime"])
    trainer = builder.build(spec=json.loads(os.environ["ZETTA_RUN_SPEC"])["trainer"])
    train_dataloader = builder.build(
        spec=json.loads(os.environ["ZETTA_RUN_SPEC"])["train_dataloader"]
    )
    val_dataloader = builder.build(spec=json.loads(os.environ["ZETTA_RUN_SPEC"])["val_dataloader"])
    lightning_train(regime, trainer, train_dataloader, val_dataloader)


@builder.register("multinode_train_launch")
@typeguard.typechecked
def multinode_train_launch(
    execution_id: str,
    num_nodes: int,
    nproc_per_node: int = 1,
    rdzv_backend: str = "c10d",
    **kwargs,  # pylint: disable=unused-argument
):
    config = torch_launcher_api.LaunchConfig(
        run_id=execution_id,
        min_nodes=num_nodes,
        max_nodes=num_nodes,
        nproc_per_node=nproc_per_node,
        rdzv_backend=rdzv_backend,
        rdzv_endpoint="master:29400" if os.environ.get("MY_ROLE") else "localhost:29400",
    )
    torch_launcher_api.elastic_launch(config, _parse_spec_and_train)()


def _get_tolerations(role: str) -> List[k8s_client.V1Toleration]:
    gpu = k8s_client.V1Toleration(
        key="nvidia.com/gpu", operator="Equal", value="present", effect="NoSchedule"
    )
    worker = k8s_client.V1Toleration(
        key=f"{role}-pool", operator="Equal", value="true", effect="NoSchedule"
    )
    return [gpu, worker]


def _spec_configmap_vol_and_ctx(
    execution_id: str,
    cluster_info: resource_allocation.k8s.ClusterInfo,
    specs: Dict[str, Any],
):
    configmap = resource_allocation.k8s.get_configmap(
        name=execution_id,
        data={f"{spec_name}.cue": json.dumps(spec) for spec_name, spec in specs.items()},
    )

    configmap_projection = k8s_client.V1ConfigMapProjection(
        name=execution_id,
        items=[k8s_client.V1KeyToPath(key=f"{spec}.cue", path=f"{spec}.cue") for spec in specs],
    )

    projected_source = k8s_client.V1ProjectedVolumeSource(
        sources=[k8s_client.V1VolumeProjection(config_map=configmap_projection)]
    )
    specs_vol = k8s_client.V1Volume(name="specs", projected=projected_source)

    specs_mount = k8s_client.V1VolumeMount(
        name="specs",
        mount_path="/opt/zetta_utils/specs",
    )

    ctx = resource_allocation.k8s.configmap_ctx_manager(
        execution_id=execution_id,
        cluster_info=cluster_info,
        configmap=configmap,
    )
    return (specs_vol, specs_mount, ctx)


def _create_ddp_master_job(
    execution_id: str,
    *,
    cluster_info: resource_allocation.k8s.ClusterInfo,
    image: str,
    resources: dict,
    train_spec: dict,
    num_nodes: int,
    env_vars: Optional[Dict[str, str]] = None,
    follow_logs: Optional[bool] = False,
    host_network: Optional[bool] = False,
):  # pylint: disable=too-many-locals
    zetta_cmd = "zetta run specs/train.cue"
    env_vars = env_vars or {}
    train_spec["@type"] = "multinode_train_launch"
    train_spec["execution_id"] = execution_id
    train_spec["num_nodes"] = num_nodes
    specs = {"train": train_spec}
    vol, mount, spec_ctx = _spec_configmap_vol_and_ctx(execution_id, cluster_info, specs)
    secrets, env_secret_mapping = resource_allocation.k8s.get_secrets_and_mapping(
        execution_id, REQUIRED_ENV_VARS
    )

    dshm = k8s_client.V1Volume(
        name="dshm", empty_dir=k8s_client.V1EmptyDirVolumeSource(medium="Memory")
    )
    tmp = k8s_client.V1Volume(
        name="tmp", empty_dir=k8s_client.V1EmptyDirVolumeSource(medium="Memory")
    )
    volumes = [vol, dshm, tmp]
    mounts = [
        mount,
        k8s_client.V1VolumeMount(mount_path="/dev/shm", name="dshm"),
        k8s_client.V1VolumeMount(mount_path="/tmp", name="tmp"),
    ]

    envs = []
    for key, val in env_vars.items():
        envs.append(k8s_client.V1EnvVar(name=key, value=val))

    ip_env = k8s_client.V1EnvVar(
        name="NODE_IP",
        value_from=k8s_client.V1EnvVarSource(
            field_ref=k8s_client.V1ObjectFieldSelector(field_path="status.hostIP")
        ),
    )

    train_pod_spec = resource_allocation.k8s.get_pod_spec(
        name=execution_id,
        image=image,
        command=["/bin/bash"],
        command_args=["-c", zetta_cmd],
        envs=envs + [ip_env],
        env_secret_mapping=env_secret_mapping,
        hostname="master",
        host_network=host_network,
        resources=resources,
        restart_policy="Never",
        # with multinode ddp we need a node on standard pool for an IP
        # that remains the same for the duration of training
        # these pools must have `master-pool=true` taint
        # not ncessary for single node ddp so it can be scheduled on preemptibles
        tolerations=_get_tolerations(role="master" if num_nodes > 1 else "worker"),
        volumes=volumes,
        volume_mounts=mounts,
    )

    train_job = resource_allocation.k8s.get_job(execution_id, pod_spec=train_pod_spec)
    train_job_ctx = resource_allocation.k8s.job_ctx_manager(
        execution_id=execution_id,
        cluster_info=cluster_info,
        job=train_job,
        secrets=secrets,
    )

    with ExitStack() as stack:
        stack.enter_context(execution_tracker.heartbeat_tracking_ctx_mngr(execution_id))
        stack.enter_context(spec_ctx)
        stack.enter_context(train_job_ctx)

        if num_nodes > 1:
            train_pod = resource_allocation.k8s.get_job_pod(train_job, cluster_info)
            aliases = [k8s_client.V1HostAlias(hostnames=["master"], ip=train_pod.status.host_ip)]
            worker_role_env = k8s_client.V1EnvVar(name="MY_ROLE", value="worker")

            worker_pod_spec = resource_allocation.k8s.get_pod_spec(
                name="workers",
                image=image,
                command=["/bin/bash"],
                command_args=["-c", zetta_cmd],
                envs=envs + [worker_role_env],
                env_secret_mapping=env_secret_mapping,
                host_network=True,
                host_aliases=aliases,
                resources=resources,
                tolerations=_get_tolerations(role="worker"),
                volumes=volumes,
                volume_mounts=mounts,
            )

            worker_deployment = resource_allocation.k8s.get_deployment(
                name=f"{execution_id}-workers",
                pod_spec=worker_pod_spec,
                replicas=num_nodes - 1,
            )

            workers_ctx = resource_allocation.k8s.deployment_ctx_mngr(
                execution_id=execution_id,
                cluster_info=cluster_info,
                deployment=worker_deployment,
                secrets=[],
            )
            stack.enter_context(workers_ctx)

        if follow_logs:
            resource_allocation.k8s.follow_job_logs(train_job, cluster_info)
        else:
            resource_allocation.k8s.wait_for_job_completion(train_job, cluster_info)


@builder.register("lightning_train_remote")
@typeguard.typechecked
def lightning_train_remote(
    worker_image: str,
    worker_resources: dict,
    spec_path: str,
    num_nodes: int = 1,
    env_vars: Optional[Dict[str, str]] = None,
    worker_cluster_name: Optional[str] = None,
    worker_cluster_region: Optional[str] = None,
    worker_cluster_project: Optional[str] = None,
    follow_logs: Optional[bool] = False,
) -> None:
    cluster_info = resource_allocation.k8s.parse_cluster_info(
        cluster_name=worker_cluster_name,
        cluster_region=worker_cluster_region,
        cluster_project=worker_cluster_project,
    )

    execution_id = mazepa.id_generation.get_unique_id(
        prefix="exec", slug_len=4, add_uuid=False, max_len=50
    )
    spec = parsing.cue.load(spec_path)

    _create_ddp_master_job(
        execution_id,
        cluster_info=cluster_info,
        env_vars=env_vars,
        follow_logs=follow_logs,
        image=worker_image,
        resources=worker_resources,
        train_spec=spec,
        num_nodes=num_nodes,
        host_network=num_nodes > 1,
    )
