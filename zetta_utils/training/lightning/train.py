from __future__ import annotations

import os
from contextlib import ExitStack
from typing import Any, Dict, Final, List, Literal, Optional

import pytorch_lightning as pl
import torch
import typeguard
from lightning_fabric.utilities.cloud_io import get_filesystem
from pytorch_lightning.strategies import ddp
from torch.distributed.launcher import api as torch_launcher_api

from kubernetes import client as k8s_client
from zetta_utils import builder, load_all_modules, log, run
from zetta_utils.cloud_management import resource_allocation
from zetta_utils.parsing import json

logger = log.get_logger("zetta_utils")

builder.register("pl.Trainer")(pl.Trainer)
builder.register("pl.callbacks.ModelCheckpoint")(pl.callbacks.ModelCheckpoint)
builder.register("pl.DDPStrategy")(ddp.DDPStrategy)

REQUIRED_ENV_VARS: Final = [
    "ZETTA_USER",
    "ZETTA_PROJECT",
    "WANDB_API_KEY",
]


def distributed_available() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


@builder.register("lightning_train", allow_parallel=False)
@typeguard.typechecked
def lightning_train(
    regime: pl.LightningModule | dict[str, Any],
    trainer: pl.Trainer | dict[str, Any],
    train_dataloader: torch.utils.data.DataLoader | dict[str, Any],
    val_dataloader: Optional[torch.utils.data.DataLoader | dict[str, Any]] = None,
    full_state_ckpt_path: str = "last",
    num_nodes: int = 1,
    retry_count: int = 3,
    local_run: bool = True,
    follow_logs: bool = False,
    image: Optional[str] = None,
    cluster_name: Optional[str] = None,
    cluster_region: Optional[str] = None,
    cluster_project: Optional[str] = None,
    env_vars: Optional[Dict[str, str]] = None,
    resource_limits: Optional[dict[str, int | float | str]] = None,
    resource_requests: Optional[dict[str, int | float | str]] = None,
    provisioning_model: Literal["standard", "spot"] = "spot",
) -> None:
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
    :param num_nodes: Number of GPU nodes for distributed training.
    :param retry_count: Max retry count for the master train job;
        excludes failures due to pod distruptions.
    :param local_run: If True run the training locally.
    :param follow_logs: If True, eagerly print logs from the pod.
        If False, will wait until job completes successfully.
    :param image: Container image to use.
    :param cluster_name: Cluster configuration.
    :param cluster_region: Cluster configuration.
    :param cluster_project: Cluster configuration.
    :param env_vars: Custom env variables to be set on pods.
    :param resource_limits: K8s reource limits per pod.
    :param resource_requests: K8s resource requests per pod.
    :param provisioning_model: VM provision type to use for worker pods.
    """
    args_mapping = {
        "regime": regime,
        "trainer": trainer,
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
    }

    if local_run:
        _lightning_train_local(
            regime=regime if not isinstance(regime, dict) else builder.build(regime),
            trainer=trainer if not isinstance(trainer, dict) else builder.build(trainer),
            train_dataloader=train_dataloader
            if not isinstance(train_dataloader, dict)
            else builder.build(train_dataloader, parallel=builder.PARALLEL_BUILD_ALLOWED),
            val_dataloader=val_dataloader
            if not isinstance(val_dataloader, dict)
            else builder.build(val_dataloader, parallel=builder.PARALLEL_BUILD_ALLOWED),
            full_state_ckpt_path=full_state_ckpt_path,
        )
        return

    if image is None:
        raise ValueError("Must provide a container image for remote training.")
    if resource_limits is None:
        raise ValueError("Must provide resource limits for remote training.")

    assert resource_allocation.gcloud.check_image_exists(image), image

    cluster_info = resource_allocation.k8s.parse_cluster_info(
        cluster_name=cluster_name,
        cluster_region=cluster_region,
        cluster_project=cluster_project,
    )
    run.register_clusters([cluster_info])

    args_mapping = {
        "regime": regime,
        "trainer": trainer,
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
    }

    train_args: dict = {
        "full_state_ckpt_path": full_state_ckpt_path,
    }

    for k, v in args_mapping.items():
        if isinstance(v, dict):
            # argument given as spec, use it directly
            train_args[k] = v
        else:
            arg_spec = builder.get_initial_builder_spec(v)
            if arg_spec is None:
                raise RuntimeError(
                    f"No builder spec found for `{k}`. Remote training requires arguments to "
                    "be created using `builder` module."
                )
            train_args[k] = arg_spec

    _lightning_train_remote(
        cluster_info=cluster_info,
        image=image,
        num_nodes=num_nodes,
        retry_count=retry_count,
        train_args=train_args,
        env_vars=env_vars,
        follow_logs=follow_logs,
        host_network=num_nodes > 1,
        resource_limits=resource_limits,
        resource_requests=resource_requests,
        provisioning_model=provisioning_model,
    )


@builder.register("_multinode_train_launch")
@typeguard.typechecked
def _multinode_train_launch(
    run_id: str,
    num_nodes: int,
    nproc_per_node: int,
    rdzv_backend: str = "c10d",
    **kwargs,  # pylint: disable=unused-argument
):
    # worker pods have MY_ROLE env set to `worker`
    is_worker = os.environ.get("MY_ROLE") == "worker"
    config = torch_launcher_api.LaunchConfig(
        run_id=run_id,
        min_nodes=num_nodes,
        max_nodes=num_nodes,
        max_restarts=1,
        nproc_per_node=nproc_per_node,
        rdzv_backend=rdzv_backend,
        rdzv_endpoint="master:29400" if is_worker else "localhost:29400",
    )
    torch_launcher_api.elastic_launch(config, _parse_spec_and_train)()


@builder.register("_lightning_train_local")
@typeguard.typechecked
def _lightning_train_local(
    regime: pl.LightningModule,
    trainer: pl.Trainer,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader | None = None,
    full_state_ckpt_path: str = "last",
):
    logger.info("Starting training...")
    if "CURRENT_BUILD_SPEC" in os.environ:
        if hasattr(trainer, "log_config"):
            trainer.log_config(json.loads(os.environ["CURRENT_BUILD_SPEC"]))
        else:
            logger.warning("Incompatible custom trainer used: Unable to save configuration.")
    else:
        logger.warning("Invoked without builder: Unable to save configuration.")

    if full_state_ckpt_path == "last":
        if get_filesystem(trainer.ckpt_path).exists(trainer.ckpt_path):
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

    train_args = None
    with open(os.environ["ZETTA_RUN_SPEC_PATH"], "r", encoding="utf-8") as f:
        train_args = json.load(f)
    logger.info(train_args)

    regime = builder.build(spec=train_args["regime"])
    trainer = builder.build(spec=train_args["trainer"])
    train_dataloader = builder.build(spec=train_args["train_dataloader"])
    try:
        val_dataloader = builder.build(spec=train_args["val_dataloader"])
    except KeyError:
        val_dataloader = None
    try:
        full_state_ckpt_path = train_args["full_state_ckpt_path"]
    except KeyError:
        full_state_ckpt_path = "last"
    _lightning_train_local(regime, trainer, train_dataloader, val_dataloader, full_state_ckpt_path)


def _get_tolerations() -> List[k8s_client.V1Toleration]:
    gpu = k8s_client.V1Toleration(
        key="nvidia.com/gpu", operator="Equal", value="present", effect="NoSchedule"
    )
    worker = k8s_client.V1Toleration(
        key="worker-pool", operator="Equal", value="true", effect="NoSchedule"
    )
    return [gpu, worker]


def _spec_configmap_vol_and_ctx(
    cluster_info: resource_allocation.k8s.ClusterInfo,
    specs: Dict[str, Any],
):
    assert run.RUN_ID is not None
    configmap = resource_allocation.k8s.get_configmap(
        name=run.RUN_ID,
        data={f"{spec_name}.cue": json.dumps(spec) for spec_name, spec in specs.items()},
    )

    configmap_projection = k8s_client.V1ConfigMapProjection(
        name=run.RUN_ID,
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
        run_id=run.RUN_ID,
        cluster_info=cluster_info,
        configmap=configmap,
    )
    return (specs_vol, specs_mount, ctx)


def _lightning_train_remote(
    *,
    cluster_info: resource_allocation.k8s.ClusterInfo,
    image: str,
    num_nodes: int,
    retry_count: int,
    train_args: dict,
    env_vars: Optional[Dict[str, str]] = None,
    follow_logs: Optional[bool] = False,
    host_network: Optional[bool] = False,
    resource_limits: Optional[dict[str, int | float | str]] = None,
    resource_requests: Optional[dict[str, int | float | str]] = None,
    provisioning_model: Literal["standard", "spot"] = "spot",
):  # pylint: disable=too-many-locals,too-many-statements
    """
    Parse spec and launch single/multinode training accordingly.
    Creates a volume mount for `train.cue` in `/opt/zetta_utils/specs`.
    Runs the command `zetta run specs/train.cue` on one or more worker pods.
    """
    assert run.RUN_ID is not None

    if train_args["trainer"]["accelerator"] in ("gpu", "cuda", "auto"):
        num_devices = int(resource_limits["nvidia.com/gpu"])  # type: ignore
        trainer_devices = train_args["trainer"]["devices"]
        if (
            isinstance(trainer_devices, int)
            and trainer_devices != -1
            and trainer_devices != num_devices
        ):
            raise ValueError(
                f"Trainer specification uses {trainer_devices} devices, "
                f"while `nvidia.com/gpu` limit is {num_devices}."
            )
    else:
        raise NotImplementedError()

    # with multinode ddp we need a node on standard pool for an IP
    # that remains the same for the duration of training
    node_selector = {"cloud.google.com/gke-provisioning": "standard"}
    if num_nodes > 1:
        train_args["run_id"] = run.RUN_ID
        train_args["num_nodes"] = num_nodes
        train_args["nproc_per_node"] = num_devices
        train_args["trainer"]["num_nodes"] = num_nodes
        train_spec = {"@type": "_multinode_train_launch", **train_args}
    else:
        train_spec = {"@type": "_lightning_train_local", **train_args}
        node_selector = {"cloud.google.com/gke-provisioning": provisioning_model}

    specs = {"train": train_spec}
    vol, mount, spec_ctx = _spec_configmap_vol_and_ctx(cluster_info, specs)
    secrets, env_secret_mapping = resource_allocation.k8s.get_secrets_and_mapping(
        run.RUN_ID, REQUIRED_ENV_VARS
    )

    volumes = [vol] + resource_allocation.k8s.get_common_volumes()
    mounts = [mount] + resource_allocation.k8s.get_common_volume_mounts()

    envs = []
    env_vars = env_vars or {}
    for key, val in env_vars.items():
        envs.append(k8s_client.V1EnvVar(name=key, value=val))

    ip_env = k8s_client.V1EnvVar(
        name="NODE_IP",
        value_from=k8s_client.V1EnvVarSource(
            field_ref=k8s_client.V1ObjectFieldSelector(field_path="status.hostIP")
        ),
    )
    flags = ""
    if builder.PARALLEL_BUILD_ALLOWED:
        flags += " -p"
    train_pod_spec = resource_allocation.k8s.get_pod_spec(
        name=run.RUN_ID,
        image=image,
        command=["/bin/bash"],
        command_args=["-c", f"zetta run {flags} specs/train.cue"],
        envs=envs + [ip_env],
        env_secret_mapping=env_secret_mapping,
        hostname="master",
        host_network=host_network,
        resources=resource_limits,
        restart_policy="Never",
        node_selector=node_selector,
        tolerations=_get_tolerations(),
        volumes=volumes,
        volume_mounts=mounts,
        resource_requests=resource_requests,
    )

    train_job_failure_policy = k8s_client.V1PodFailurePolicy(
        rules=[
            k8s_client.V1PodFailurePolicyRule(
                action="Ignore",
                on_pod_conditions=[
                    k8s_client.V1PodFailurePolicyOnPodConditionsPattern(
                        status="True", type="DisruptionTarget"
                    )
                ],
            )
        ]
    )
    train_job = resource_allocation.k8s.get_job(
        name=run.RUN_ID,
        pod_spec=train_pod_spec,
        backoff_limit=retry_count,
        pod_failure_policy=train_job_failure_policy,
    )
    train_job_ctx = resource_allocation.k8s.job_ctx_manager(
        run_id=run.RUN_ID,
        cluster_info=cluster_info,
        job=train_job,
        secrets=secrets,
    )

    with ExitStack() as stack:
        stack.enter_context(spec_ctx)
        stack.enter_context(train_job_ctx)

        if num_nodes > 1:
            train_pod = resource_allocation.k8s.get_job_pod(train_job, cluster_info)
            aliases = [k8s_client.V1HostAlias(hostnames=["master"], ip=train_pod.status.host_ip)]
            worker_env = [
                k8s_client.V1EnvVar(name="MY_ROLE", value="worker"),
                k8s_client.V1EnvVar(name="MASTER_ADDR", value="master"),
            ]

            flags += " --no-main-run-process"
            worker_pod_spec = resource_allocation.k8s.get_pod_spec(
                name="workers",
                image=image,
                command=["/bin/bash"],
                command_args=["-c", f"zetta run -r {run.RUN_ID} {flags} specs/train.cue"],
                envs=envs + worker_env,
                env_secret_mapping=env_secret_mapping,
                host_network=True,
                host_aliases=aliases,
                resources=resource_limits,
                node_selector={"cloud.google.com/gke-provisioning": provisioning_model},
                tolerations=_get_tolerations(),
                volumes=volumes,
                volume_mounts=mounts,
                resource_requests=resource_requests,
            )

            worker_deployment = resource_allocation.k8s.get_deployment(
                name=f"{run.RUN_ID}-workers",
                pod_spec=worker_pod_spec,
                replicas=num_nodes - 1,
            )

            workers_ctx = resource_allocation.k8s.deployment_ctx_mngr(
                run_id=run.RUN_ID,
                cluster_info=cluster_info,
                deployment=worker_deployment,
                secrets=[],
            )
            stack.enter_context(workers_ctx)

        if follow_logs:
            resource_allocation.k8s.follow_job_logs(train_job, cluster_info)
        else:
            resource_allocation.k8s.wait_for_job_completion(train_job, cluster_info)
