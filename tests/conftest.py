import os
import time
from unittest import mock

import pytest

import docker


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
        # ports={"8081": "8081"},
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
        raise RuntimeError(f"Container failed to start: {container.logs()}")

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
