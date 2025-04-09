import random
import time

import fsspec
import torch

from zetta_utils import builder, log, mazepa
from zetta_utils.mazepa.tasks import taskable_operation

logger = log.get_logger("zetta_utils")


@builder.register("test_gcs_access")
def test_gcs_access(read_path: str, write_path):
    with fsspec.open(read_path, "r") as f:
        data_read_x0 = f.read()

    with fsspec.open(write_path, "w") as f:
        f.write(data_read_x0)

    with fsspec.open(write_path, "r") as f:
        data_read_x1 = f.read()

    assert data_read_x1 == data_read_x0


@taskable_operation
def dummy_task(num_seconds: int) -> int:
    logger.info(f"Sleeping for {num_seconds}s...")
    time.sleep(num_seconds)
    return num_seconds


@builder.register("keda_test_flow")
@mazepa.flow_schema
def keda_test_flow(num_tasks: int):
    for _ in range(num_tasks):
        yield dummy_task.make_task(random.randint(0, 10))
    yield dummy_task.make_task(random.randint(600, 1200))


@taskable_operation
def dummy_cpu_task(num_seconds: int) -> bool:
    logger.info("CPU Task.")
    time.sleep(num_seconds)
    return False


@taskable_operation
def dummy_gpu_task(num_seconds: int) -> bool:
    logger.info(f"GPU Task. {torch.cuda.device_count()} available.")
    time.sleep(num_seconds)
    return True


@builder.register("group_test_flow")
@mazepa.flow_schema
def group_test_flow(num_tasks: int, type0: str, type1: str):
    for _ in range(num_tasks):
        task = dummy_cpu_task.make_task(random.randint(0, 3))
        task.worker_type = type0
        yield task

    for _ in range(num_tasks):
        task = dummy_gpu_task.make_task(random.randint(0, 3))
        task.worker_type = type1
        yield task
