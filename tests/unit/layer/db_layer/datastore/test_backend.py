# pylint: disable=redefined-outer-name

import os
import time
from unittest import mock

import pytest

import docker
from zetta_utils.layer.db_layer.datastore import DatastoreBackend, build_datastore_layer


@pytest.fixture(scope="session")
def datastore_emulator():
    """Ensure that the DataStore service is up and responsive."""

    client = docker.from_env()
    project = "test-project"
    options = "--no-store-on-disk --consistency=1.0 --host-port=0.0.0.0:8081"
    command = f"gcloud beta emulators datastore start {options}"

    container = client.containers.run(
        "motemen/datastore-emulator",
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
        container.reload()

    endpoint = "127.0.0.1:8081"

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


def test_build_layer(datastore_emulator):
    layer = build_datastore_layer(datastore_emulator, datastore_emulator)
    assert isinstance(layer.backend, DatastoreBackend)


def test_write_scalar(datastore_emulator) -> None:
    layer = build_datastore_layer(datastore_emulator, datastore_emulator)
    layer["key"] = "val"
    assert layer["key"] == "val"
