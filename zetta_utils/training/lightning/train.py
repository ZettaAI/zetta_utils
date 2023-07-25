from __future__ import annotations

import json
import os
from contextlib import ExitStack
from typing import Final

import pytorch_lightning as pl
import torch
import typeguard
from pytorch_lightning.strategies import ddp
from pytorch_lightning.utilities.cloud_io import get_filesystem

from kubernetes import client as k8s_client  # type: ignore
from zetta_utils import builder, log, parsing
from zetta_utils.cloud import resource_allocation

DEFAULT_GCP_CLUSTER_NAME: Final = "zutils-x3"
DEFAULT_GCP_CLUSTER_REGION: Final = "us-east1"
DEFAULT_GCP_CLUSTER_PROJECT: Final = "zetta-research"

DEFAULT_GCP_CLUSTER: Final = resource_allocation.k8s.ClusterInfo(
    name=DEFAULT_GCP_CLUSTER_NAME,
    region=DEFAULT_GCP_CLUSTER_REGION,
    project=DEFAULT_GCP_CLUSTER_PROJECT,
)

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


@builder.register("lightning_train_remote")
@typeguard.typechecked
def lightning_train_remote(image: str, resources: dict, spec_path: str) -> None:
    execution_id = "test"
    spec = parsing.cue.load(spec_path)
    configmap = resource_allocation.k8s.get_configmap(
        name=execution_id, data={"spec.cue": json.dumps(spec)}
    )
    spec_source = k8s_client.V1ConfigMapVolumeSource(
        name=execution_id, items=[k8s_client.V1KeyToPath(key="spec.cue", path="spec.cue")]
    )
    spec_vol = k8s_client.V1Volume(name="spec", config_map=spec_source)
    spec_vol_mount = k8s_client.V1VolumeMount(
        name="spec", mount_path="/opt/zetta_utils/spec.cue", sub_path="spec.cue"
    )

    command = ["zetta"]
    command_args = ["run", "spec.cue"]

    worker = k8s_client.V1Toleration(
        key="worker-pool", operator="Equal", value="true", effect="NoSchedule"
    )

    gpu = k8s_client.V1Toleration(
        key="nvidia.com/gpu", operator="Equal", value="present", effect="NoSchedule"
    )

    secrets, env_secret_mapping = resource_allocation.k8s.get_secrets_and_mapping(
        execution_id, REQUIRED_ENV_VARS
    )
    train_pod_spec = resource_allocation.k8s.get_pod_spec(
        name=execution_id,
        image=image,
        command=command,
        command_args=command_args,
        env_secret_mapping=env_secret_mapping,
        resources=resources,
        restart_policy="Never",
        tolerations=[worker, gpu],
        volumes=[spec_vol],
        volume_mounts=[spec_vol_mount],
    )

    train_job = resource_allocation.k8s.get_job(execution_id, pod_spec=train_pod_spec)

    configmap_ctx = resource_allocation.k8s.configmap_ctx_manager(
        execution_id=execution_id, cluster_info=DEFAULT_GCP_CLUSTER, configmap=configmap
    )

    train_job_ctx = resource_allocation.k8s.job_ctx_manager(
        execution_id=execution_id,
        cluster_info=DEFAULT_GCP_CLUSTER,
        job=train_job,
        secrets=secrets,
    )

    with ExitStack() as stack:
        stack.enter_context(configmap_ctx)
        stack.enter_context(train_job_ctx)
