# pylint: disable=redefined-outer-name

import pickle
from typing import cast

import pytest

from zetta_utils.layer.db_layer import DBDataT
from zetta_utils.layer.db_layer.datastore import DatastoreBackend, build_datastore_layer


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

    data_user: DBDataT = [
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


def test_pickle(datastore_emulator) -> None:
    layer = build_datastore_layer(datastore_emulator, datastore_emulator)
    layer_backend = cast(DatastoreBackend, layer.backend)
    assert layer_backend.client is not None  # ensure client is initialized

    layer2 = pickle.loads(pickle.dumps(layer))
    layer_backend2 = cast(DatastoreBackend, layer2.backend)
    assert layer_backend2.project == layer_backend.project
    assert layer_backend2.namespace == layer_backend.namespace
    assert layer_backend2.exclude_from_indexes == layer_backend.exclude_from_indexes
    assert layer_backend2.name == layer_backend.name
