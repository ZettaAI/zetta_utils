import random
import time

import fsspec
import gcsfs
import tensorstore as ts
import torch
from cloudfiles import CloudFiles
from cloudvolume import CloudVolume

from zetta_utils import builder, log, mazepa
from zetta_utils.mazepa import semaphore
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


@taskable_operation
def instrumentation_test_task(
    read_path: str,
    write_path: str,
    num_iterations: int = 10,
    work_seconds: float = 1.0,
):
    """Exercises all three sidecar instrumentation channels in one task.

    - GCS stats: fsspec read + fsspec write on each iteration (Class B + Class A).
    - Semaphore stats: acquires read/cpu/cuda/write semaphores with realistic
      wait+lease times. Contention (wait > 0) comes from num_procs > the
      semaphore width at the pool level; no in-task multiprocessing needed.
    - Resource stats: CPU-bound inner loop (not sleep) so ResourceMonitor's
      psutil samples register real CPU utilization.
    """
    logger.info(
        f"Starting instrumentation test task: "
        f"{num_iterations} iters, {work_seconds}s CPU per iter"
    )
    for i in range(num_iterations):
        with semaphore("read"):
            with fsspec.open(read_path, "r") as f:
                data = f.read()

        with semaphore("cpu"):
            # Real CPU burn so ResourceMonitor sees non-idle pod CPU.
            end = time.time() + work_seconds
            acc = 0
            while time.time() < end:
                acc += sum(j * j for j in range(2000))
            logger.debug(f"iter {i} acc={acc}")

        with semaphore("cuda"):
            # Brief hold -- we don't need real GPU work to exercise the
            # cuda semaphore's TimingTracker.
            time.sleep(0.05)

        with semaphore("write"):
            with fsspec.open(f"{write_path}_instr_{i}", "w") as f:
                f.write(data)


@builder.register("instrumentation_test_flow")
@mazepa.flow_schema
def instrumentation_test_flow(
    num_tasks: int,
    read_path: str,
    write_path: str,
    num_iterations: int = 10,
    work_seconds: float = 1.0,
):
    """Fan out N instrumentation tasks to stress the worker pool semaphore
    contention.

    Produces roughly:
    - gcs_stats:       num_tasks * num_iterations * 2 requests (read+write)
    - semaphore_stats: num_tasks * num_iterations acquisitions per type
    - resource_stats:  CPU spikes during the work loop; mem/disk/net samples
    """
    # Seed the read path so worker tasks have something to read. Idempotent:
    # overwrites on every run. Kept tiny to keep egress costs negligible.
    logger.info(f"Seeding instrumentation test input at {read_path}")
    with fsspec.open(read_path, "w") as f:
        f.write("instrumentation test seed\n")

    for _ in range(num_tasks):
        yield instrumentation_test_task.make_task(
            read_path, write_path, num_iterations, work_seconds
        )


@taskable_operation
def cost_tracking_test_task(
    bucket_prefix: str,
    pod_subprefix: str,
    num_chunks: int = 20,
    chunk_kb: int = 4,
):
    """Exercise the GCS classifier across all the API shapes mproxy sees.

    Generates traffic that hits every classifier branch:
    - XML PUT (cf.put)         → Class A insert
    - XML GET (cf.get)         → Class B get  + egress
    - XML HEAD (cf.exists)     → Class B get_metadata
    - JSON list (cf.list)      → Class A list_objects
    - Batch delete (cf.delete) → /batch/storage/v1 → _batch (uncounted)

    After the flow completes, inspect the per-pod stats in Firestore
    (POD_STATS_DB) for the run and verify:
    - bucket attribution maps to the real bucket name (no "_unknown")
    - "_unclassified" count is zero (no classifier-coverage gaps)
    - "_batch" count > 0 (batch deletes ran)
    - class_a_count, class_b_count, egress_bytes are non-zero
    """
    base = f"{bucket_prefix.rstrip('/')}/{pod_subprefix}"
    cf = CloudFiles(base)
    payload = b"x" * (chunk_kb * 1024)
    keys = [f"chunk_{i}" for i in range(num_chunks)]

    logger.info(f"cost_tracking_test_task: writing {num_chunks} chunks under {base}")
    cf.puts([(k, payload) for k in keys])

    logger.info("cost_tracking_test_task: reading chunks back")
    for k in keys:
        _ = cf.get(k)

    logger.info("cost_tracking_test_task: HEAD on each chunk")
    for k in keys:
        cf.exists(k)

    logger.info("cost_tracking_test_task: listing prefix")
    listed = list(cf.list())
    assert len(listed) >= num_chunks, f"expected >= {num_chunks} listed, got {len(listed)}"

    logger.info("cost_tracking_test_task: batch delete")
    cf.delete(keys)


@builder.register("cost_tracking_test_flow")
@mazepa.flow_schema
def cost_tracking_test_flow(
    num_tasks: int,
    bucket_prefix: str,
    num_chunks: int = 20,
    chunk_kb: int = 4,
):
    """Fan out N cost-tracking tasks under per-task subprefixes.

    Each task uses a unique subprefix so concurrent pods don't fight over
    the same keys. Roughly per task:
    - num_chunks Class A inserts, Class B gets, Class B head
    - 1 Class A list_objects
    - 1 _batch entry (one batch HTTP request carrying num_chunks deletes)
    """
    for i in range(num_tasks):
        sub = f"task_{int(time.time())}_{random.randint(0, 1_000_000):07d}_{i}"
        yield cost_tracking_test_task.make_task(bucket_prefix, sub, num_chunks, chunk_kb)
