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


def test_build_layer(datastore_emulator):
    layer = build_datastore_layer(datastore_emulator, datastore_emulator)
    assert isinstance(layer.backend, DatastoreBackend)


def test_read_write_simple(datastore_emulator) -> None:
    layer = build_datastore_layer(datastore_emulator, datastore_emulator)

    # non existing key
    with pytest.raises(KeyError):
        _ = layer["key"]

    layer["key"] = "val"
    assert layer["key"] == "val"

    parent_key = layer.backend.client.key("Row", "key")  # type: ignore
    child_key = layer.backend.client.key("Column", "value", parent=parent_key)  # type: ignore

    entity = layer.backend.client.get(child_key)  # type: ignore
    assert entity["value"] == "val"


def test_read_write(datastore_emulator) -> None:
    layer = build_datastore_layer(datastore_emulator, datastore_emulator)

    row_keys = ["key0", "key1"]
    idx_user = (row_keys, ("col0", "col1"))

    data_user = [
        {"col0": "val0", "col1": "val1"},
        {"col0": "val0"},
    ]

    layer[idx_user] = data_user

    data = layer[idx_user]
    assert data == data_user


def test_with_changes(datastore_emulator) -> None:
    backend = DatastoreBackend(datastore_emulator, project=datastore_emulator)
    backend2 = backend.with_changes(namespace=backend.namespace, project=backend.project)
    assert isinstance(backend2, DatastoreBackend)

    with pytest.raises(KeyError):
        backend2 = backend.with_changes(
            namespace=backend.namespace,
            project=backend.project,
            some_key="test",
        )
