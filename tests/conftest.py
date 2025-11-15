# Print immediately to stderr before ANY imports
import sys
print("=" * 80, file=sys.stderr, flush=True)
print("DEBUG: conftest.py started - VERY FIRST LINE", file=sys.stderr, flush=True)
print(f"DEBUG: Python version: {sys.version}", file=sys.stderr, flush=True)
print(f"DEBUG: sys.executable: {sys.executable}", file=sys.stderr, flush=True)
print("=" * 80, file=sys.stderr, flush=True)

import os

# Set environment variable
os.environ.setdefault('TZ', 'UTC')
print("DEBUG: Set TZ environment variable", file=sys.stderr, flush=True)

# Check if anything has already imported tensorstore
print(f"DEBUG: 'tensorstore' in sys.modules: {'tensorstore' in sys.modules}", file=sys.stderr, flush=True)
print(f"DEBUG: 'neuroglancer' in sys.modules: {'neuroglancer' in sys.modules}", file=sys.stderr, flush=True)

# CRITICAL: Import tensorstore FIRST to avoid abseil initialization conflicts
# This must be the very first import before anything else
print("DEBUG: About to import tensorstore...", file=sys.stderr, flush=True)
try:
    import tensorstore  # noqa: F401
    print("DEBUG: tensorstore module imported", file=sys.stderr, flush=True)
    print(f"DEBUG: tensorstore.__file__: {tensorstore.__file__}", file=sys.stderr, flush=True)

    # Force load C++ extension
    print("DEBUG: About to access tensorstore._tensorstore...", file=sys.stderr, flush=True)
    _ = tensorstore._tensorstore
    print("DEBUG: tensorstore._tensorstore accessed successfully", file=sys.stderr, flush=True)
except (ImportError, AttributeError) as e:
    print(f"DEBUG: Tensorstore import/access failed: {e}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc(file=sys.stderr)

# Import neuroglancer
print("DEBUG: About to import neuroglancer...", file=sys.stderr, flush=True)
try:
    import neuroglancer  # noqa: F401
    print("DEBUG: neuroglancer imported successfully", file=sys.stderr, flush=True)
    print(f"DEBUG: neuroglancer.__file__: {neuroglancer.__file__}", file=sys.stderr, flush=True)
except ImportError as e:
    print(f"DEBUG: Neuroglancer import failed: {e}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc(file=sys.stderr)

print("DEBUG: About to import other modules (time, mock, pytest, docker)...", file=sys.stderr, flush=True)

import time
from unittest import mock

import pytest

import docker

print("DEBUG: About to import from zetta_utils...", file=sys.stderr, flush=True)
from zetta_utils import constants
print("DEBUG: Successfully imported zetta_utils.constants", file=sys.stderr, flush=True)

constants.RUN_DATABASE = None

print("=" * 80, file=sys.stderr, flush=True)
print("DEBUG: conftest.py module-level imports completed successfully", file=sys.stderr, flush=True)
print("=" * 80, file=sys.stderr, flush=True)


def pytest_addoption(parser):
    parser.addoption("--run-integration", default=False, help="Run integration tests")


@pytest.fixture(scope="session")
def datastore_emulator():
    """Ensure that the DataStore service is up and responsive."""

    client = docker.from_env()  # type: ignore
    project = "test-project"
    options = "--no-store-on-disk --consistency=1.0 --host-port=0.0.0.0:8081"
    command = f"gcloud --project {project} beta emulators datastore start {options}"

    container = client.containers.run(
        "motemen/datastore-emulator:alpine",
        command=command,
        detach=True,
        remove=True,
        network_mode="host",
    )

    timeout = 120
    stop_time = 1
    elapsed_time = 0
    while container.status != "running" and elapsed_time < timeout:
        time.sleep(stop_time)
        elapsed_time += stop_time
        try:
            container.reload()
        except docker.errors.DockerException:  # type: ignore
            break

    if container.status != "running":
        raise RuntimeError(f"Datastore container failed to start: {container.logs()}")

    time.sleep(5)  # wait for emulator to boot

    endpoint = "localhost:8081"

    environment = {}
    environment["DATASTORE_EMULATOR_HOST"] = endpoint
    environment["DATASTORE_DATASET"] = project
    environment["DATASTORE_EMULATOR_HOST_PATH"] = "localhost:8081/datastore"
    environment["DATASTORE_HOST"] = f"http://{endpoint}"
    environment["DATASTORE_PROJECT_ID"] = project

    with mock.patch.dict(os.environ, environment):
        yield project

    container.kill()
    time.sleep(0.2)


@pytest.fixture(scope="session")
def firestore_emulator():
    """Ensure that the Firestore service is up and responsive."""

    client = docker.from_env()  # type: ignore
    project = "test-project"
    port = "8080"
    container = client.containers.run(
        "mtlynch/firestore-emulator:latest",
        detach=True,
        remove=True,
        network_mode="host",
        environment={"FIRESTORE_PROJECT_ID": project, "PORT": port},
    )

    timeout = 120
    stop_time = 1
    elapsed_time = 0
    while container.status != "running" and elapsed_time < timeout:
        time.sleep(stop_time)
        elapsed_time += stop_time
        try:
            container.reload()
        except docker.errors.DockerException:  # type: ignore
            break

    if container.status != "running":
        raise RuntimeError(f"Firestore container failed to start: {container.logs()}")

    time.sleep(5)  # wait for emulator to boot

    environment = {}
    environment["FIRESTORE_EMULATOR_HOST"] = f"localhost:{port}"
    environment["FIRESTORE_PROJECT_ID"] = project

    with mock.patch.dict(os.environ, environment):
        yield project

    container.kill()
    time.sleep(0.2)
