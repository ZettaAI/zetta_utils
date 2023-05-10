# pylint: disable=too-many-locals

from __future__ import annotations

import json
import os
from contextlib import ExitStack
from typing import Dict, Final, Optional

import pytorch_lightning as pl
import torch
import typeguard
from pytorch_lightning.strategies import ddp
from torch.distributed.launcher import api as torch_launcher_api

from kubernetes import client as k8s_client  # type: ignore
from zetta_utils import builder, log
from zetta_utils.cloud import resource_allocation

logger = log.get_logger("zetta_utils")

builder.register("pl.Trainer")(pl.Trainer)
builder.register("pl.callbacks.ModelCheckpoint")(pl.callbacks.ModelCheckpoint)
builder.register("pl.DDPStrategy")(ddp.DDPStrategy)
builder.register("torch.distributed.LaunchConfig")(torch_launcher_api.LaunchConfig)

REQUIRED_ENV_VARS: Final = [
    # "GRAFANA_CLOUD_ACCESS_KEY",
    "ZETTA_USER",
    "ZETTA_PROJECT",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_DEFAULT_REGION",
    "WANDB_API_KEY",
]


@builder.register("lightning_train")
@typeguard.typechecked
def lightning_train(
    regime: pl.LightningModule,
    trainer: pl.Trainer,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader | None = None,
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

    trainer.fit(
        model=regime,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=None,
    )


@builder.register("multinode_train_launch")
@typeguard.typechecked
def multinode_train_launch(
    num_nodes: int,
    rdzv_endpoint: str,
    regime: pl.LightningModule,
    trainer: pl.Trainer,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader | None = None,
    nproc_per_node: int = 1,
    rdzv_backend: str = "c10d",
):
    config = torch_launcher_api.LaunchConfig(
        run_id="test",
        min_nodes=num_nodes,
        max_nodes=num_nodes,
        nproc_per_node=nproc_per_node,
        rdzv_backend=rdzv_backend,
        rdzv_endpoint=rdzv_endpoint,
    )
    torch_launcher_api.elastic_launch(config, lightning_train)(
        regime,
        trainer,
        train_dataloader,
        val_dataloader,
    )


@builder.register("multinode_train")
@typeguard.typechecked
def multinode_train(
    execution_id: str,
    image: str,
    resources: Dict[str, int | float | str],
    master_node_ip: str,
    num_nodes: int,
    env_vars: Optional[Dict[str, str]] = None,
):
    env_vars = env_vars or {}
    command = ["zetta"]
    command_args = ["run", "ddp_master.cue"]
    resources["memory"] = "10240Mi"
    resources["nvidia.com/gpu"] = "1"

    envs = []
    for key, val in env_vars.items():
        envs.append(k8s_client.V1EnvVar(name=key, value=val))

    schedule_toleration = k8s_client.V1Toleration(
        key="nvidia.com/gpu", operator="Equal", value="present", effect="NoSchedule"
    )

    node_selector = {"cloud.google.com/gke-nodepool": "master"}

    master_pod_spec = resource_allocation.k8s.get_pod_spec(
        name="master",
        image=image,
        command=command,
        command_args=command_args,
        envs=envs,
        hostname="master",
        host_network=True,
        node_selector=node_selector,
        resources=resources,
        restart_policy="Never",
        tolerations=[schedule_toleration],
    )

    master_selector = {"app": "master"}
    ports = [k8s_client.V1ServicePort(port=29400, protocol="TCP", target_port=29400)]
    c10d_service = resource_allocation.k8s.get_service(
        "c10d", ports=ports, selector=master_selector, service_type="NodePort"
    )

    master_job = resource_allocation.k8s.get_job("master", pod_spec=master_pod_spec)

    command_args = ["run", "ddp_workers.cue"]
    node_selector = {"cloud.google.com/gke-nodepool": "workers"}

    host_aliases = [k8s_client.V1HostAlias(hostnames=["master"], ip=master_node_ip)]

    worker_pod_spec = resource_allocation.k8s.get_pod_spec(
        name="workers",
        image=image,
        command=command,
        command_args=command_args,
        envs=envs,
        host_network=True,
        host_aliases=host_aliases,
        node_selector=node_selector,
        resources=resources,
        tolerations=[schedule_toleration],
    )

    worker_deployment = resource_allocation.k8s.get_deployment(
        name="workers",
        pod_spec=worker_pod_spec,
        replicas=num_nodes - 1,
    )

    cluster_info = resource_allocation.k8s.ClusterInfo(
        "ddp-test-v0", region="us-east1", project="zetta-research"
    )

    c10d_ctx = resource_allocation.k8s.service_ctx_manager(
        execution_id=execution_id,
        cluster_info=cluster_info,
        service=c10d_service,
    )

    master_ctx = resource_allocation.k8s.job_ctx_manager(
        execution_id=execution_id,
        cluster_info=cluster_info,
        job=master_job,
    )

    secrets, _ = resource_allocation.k8s.get_secrets_and_mapping(execution_id, REQUIRED_ENV_VARS)

    workers_ctx = resource_allocation.k8s.deployment_ctx_mngr(
        execution_id=execution_id,
        cluster_info=cluster_info,
        deployment=worker_deployment,
        secrets=secrets,
    )

    with ExitStack() as stack:
        stack.enter_context(c10d_ctx)
        stack.enter_context(master_ctx)
        stack.enter_context(workers_ctx)
