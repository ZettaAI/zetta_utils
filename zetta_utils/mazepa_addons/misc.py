import random
import time

import fsspec
import gcsfs
import tensorstore as ts
import torch
from cloudfiles import CloudFiles
from cloudvolume import CloudVolume

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


@taskable_operation
def dummy_oom_trigger(num_seconds: int):
    time.sleep(num_seconds)
    memory_hog = []
    try:
        while True:
            memory_hog.append("X" * 10_000_000)
            time.sleep(0.1)
    except MemoryError:
        print("MemoryError caught (unlikely in K8s OOM kill)")


@builder.register("dummy_oom_flow")
@mazepa.flow_schema
def dummy_oom_flow(num_tasks: int, sleep_sec: tuple[int, int]):
    for _ in range(num_tasks):
        task = dummy_oom_trigger.make_task(random.randint(*sleep_sec))
        yield task


@taskable_operation
def test_gcs_access_loop(
    read_path: str,
    write_path: str,
    cv_path: str | None = None,
    num_iterations: int = 100,
):
    """Test all GCS access patterns to verify proxy tracking captures them all.

    Tests: gcsfs, fsspec, cloudfiles, tensorstore (kvstore), cloudvolume (optional)
    """
    logger.info(f"Starting GCS access test: {num_iterations} iterations")

    gcs_fs = gcsfs.GCSFileSystem()
    bucket_path = read_path.replace("gs://", "").rsplit("/", 1)[0]
    cf = CloudFiles(f"gs://{bucket_path}")
    file_name = read_path.replace("gs://", "").rsplit("/", 1)[1]

    gcs_path_parts = write_path.replace("gs://", "").split("/", 1)
    ts_bucket = gcs_path_parts[0]
    ts_base_path = gcs_path_parts[1] if len(gcs_path_parts) > 1 else ""

    # Optional: CloudVolume for precomputed volumes
    cv_vol = None
    if cv_path:
        try:
            cv_vol = CloudVolume(cv_path)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(f"Could not initialize CloudVolume: {e}")

    for i in range(num_iterations):
        logger.info(f"Iteration {i+1}/{num_iterations}")

        # 1. gcsfs - read and write (Class B read, Class A write)
        data_gcsfs = gcs_fs.cat(read_path).decode()
        gcs_fs.pipe(f"{write_path}_gcsfs_{i}", data_gcsfs.encode())

        # 2. fsspec - read and write (Class B read, Class A write)
        with fsspec.open(read_path, "r") as f:
            data_fsspec = f.read()
        with fsspec.open(f"{write_path}_fsspec_{i}", "w") as f:
            f.write(data_fsspec)

        # 3. cloudfiles - read and write (Class B read, Class A write)
        data_cf = cf.get(file_name)
        cf.put(f"{file_name}_cf_{i}", data_cf)

        # 4. tensorstore kvstore - read and write (Class B read, Class A write)
        # Uses GCS kvstore driver directly, no precomputed volume needed
        try:
            ts_kvstore = ts.KvStore.open(
                {
                    "driver": "gcs",
                    "bucket": ts_bucket,
                }
            ).result()
            ts_key = f"{ts_base_path}_ts_{i}"
            ts_kvstore.write(ts_key, data_gcsfs.encode()).result()
            _ = ts_kvstore.read(ts_key).result()
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(f"tensorstore kvstore failed: {e}")

        # 5. CloudVolume - read small chunk if available (Class B read)
        if cv_vol is not None:
            try:
                _ = cv_vol[0:1, 0:1, 0:1]
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.warning(f"CloudVolume read failed: {e}")

        time.sleep(1)
    logger.info("GCS access test complete")


@builder.register("test_gcs_access_flow")
@mazepa.flow_schema
def test_gcs_access_flow(
    num_tasks: int,
    read_paths: list[str],
    write_paths: list[str],
    cv_path: str | None = None,
    num_iterations: int = 300,
):
    for _ in range(num_tasks):
        for rp, wp in zip(read_paths, write_paths):
            task = test_gcs_access_loop.make_task(rp, wp, cv_path, num_iterations)
            yield task
