# pylint: disable=redefined-outer-name

import math
import pickle
from random import randint
from typing import cast

import pytest

from zetta_utils.layer.db_layer import DBDataT, DBLayer
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
    assert layer.get("key") == "val"
    assert layer.get("non_existent_key") is None
    assert layer.get("non_existent_key", "default_val") == "default_val"


def _write_some_data(layer: DBLayer):
    layer.clear()
    row_keys = ["key0", "key1", "key2"]
    idx_user = (row_keys, ("col0", "col1"))
    data_user: DBDataT = [
        {"col0": "val0", "col1": "val0"},
        {"col0": "val0"},
        {"col1": "val1"},
    ]
    layer[idx_user] = data_user
    return idx_user, data_user


def test_read_write(datastore_emulator) -> None:
    layer = build_datastore_layer(datastore_emulator, datastore_emulator)
    idx_user, data_user = _write_some_data(layer)
    data = layer[idx_user]
    assert data == data_user
    assert layer[("key0", "col1")] == "val0"


def test_query_and_keys(datastore_emulator) -> None:
    layer = build_datastore_layer(datastore_emulator, datastore_emulator)
    _write_some_data(layer)
    col_filter = {"col1": ["val0"]}
    result = layer.query(column_filter=col_filter, return_columns=("col0", "col1"))
    assert "key0" in result and len(result) == 1

    col_filter = {"col1": ["val1"]}
    result = layer.query(column_filter=col_filter)
    assert "key2" in result and len(result) == 1
    assert len(layer.keys()) == 3


def test_delete_and_clear(datastore_emulator) -> None:
    layer = build_datastore_layer(datastore_emulator, datastore_emulator)
    _write_some_data(layer)
    assert len(layer.keys()) == 3

    del layer["key0"]
    assert len(layer.keys()) == 2

    layer.clear()
    assert len(layer.keys()) == 0


def _test_batches(layer: DBLayer, batch_size: int, return_columns: tuple[str, ...] = ()):
    ROW_COUNT = len(layer.keys())
    batches = []
    batch_count = int(math.ceil(len(layer) / batch_size))
    for i in range(batch_count):
        batches.append(layer.get_batch(i, batch_size, return_columns=return_columns))

    batch_0 = layer.get_batch(0, batch_size)  # test with no return_columns
    assert len(batches[0]) == len(batch_0)

    batch_keys = [k for b in batches for k in b.keys()]
    batch_sizes = [len(b) for b in batches]

    assert sum(batch_sizes) == ROW_COUNT
    assert len(batch_keys) == ROW_COUNT
    assert len(set(batch_keys)) == ROW_COUNT

    # test batch_size > len(layer), must return all rows
    batch = layer.get_batch(0, ROW_COUNT + 1, return_columns=return_columns)
    assert len(batch) == ROW_COUNT

    # test out of bounds error
    with pytest.raises(IndexError):
        layer.get_batch(batch_count, batch_size)


def test_batching(datastore_emulator, mocker) -> None:
    layer = build_datastore_layer(datastore_emulator, datastore_emulator)

    ROW_COUNT = 150
    COLS = ("col_a", "col_b")
    mocker.patch(
        "zetta_utils.layer.db_layer.datastore.backend.DatastoreBackend.__len__",
        return_value=ROW_COUNT,
    )
    rows: DBDataT = [
        dict(zip(COLS, [randint(0, 1000), randint(0, 1000)])) for _ in range(1, ROW_COUNT + 1)
    ]
    row_keys = list(str(x) for x in range(1, ROW_COUNT + 1))  # cannot use 0 as key in Datastore.
    layer[(row_keys, COLS)] = rows

    assert len(layer.keys()) == ROW_COUNT

    _test_batches(layer, 75, return_columns=COLS)  # test even batch size, total divisible by batch
    _test_batches(layer, 45, return_columns=COLS)  # test odd batch size


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
