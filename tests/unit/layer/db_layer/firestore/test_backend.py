# pylint: disable=redefined-outer-name

import math
import pickle
from random import randint
from typing import cast

import pytest

from zetta_utils.layer.db_layer import DBDataT, DBLayer
from zetta_utils.layer.db_layer.firestore import FirestoreBackend, build_firestore_layer


def test_build_layer(firestore_emulator):
    layer = build_firestore_layer("test", project=firestore_emulator)
    assert isinstance(layer.backend, FirestoreBackend)


def test_read_write_simple(firestore_emulator) -> None:
    layer = build_firestore_layer("test", project=firestore_emulator)

    # non existing key
    with pytest.raises(KeyError):
        _ = layer["key"]

    layer["key"] = "val"
    assert layer["key"] == "val"

    doc_ref = layer.backend.client.collection("test").document("key")  # type: ignore
    doc = doc_ref.get().to_dict()

    assert doc["value"] == "val"
    assert layer.get("key") == "val"
    assert layer.get("non_existent_key") is None
    assert layer.get("non_existent_key", "default_val") == "default_val"


def _write_some_data(layer: DBLayer):
    layer.clear()
    row_keys = ["key0", "key1", "key2"]
    idx_user = (row_keys, ("col0", "col1", "col2", "tags"))
    data_user: DBDataT = [
        {"col0": "val0", "col1": "val0"},
        {"col0": "val0", "tags": ["tag1"]},
        {"col1": "val1", "tags": ["tag0", "tag1"]},
    ]
    layer[idx_user] = data_user
    return idx_user, data_user


def _write_some_query_data(layer: DBLayer):
    layer.clear()
    row_keys = ["key0", "key1", "key2", "key3", "key4", "key5"]
    idx_user = (row_keys, ("col0", "col1", "col2", "tags"))
    data_user: DBDataT = [
        {"col0": "val0", "col1": "val0"},
        {"col0": "val0", "tags": ["tag1"]},
        {"col1": "val1", "tags": ["tag0", "tag1"]},
        {"col2": 3},
        {"col2": 5},
        {"col2": 1},
    ]
    layer[idx_user] = data_user
    return idx_user, data_user


def test_read_write(firestore_emulator) -> None:
    layer = build_firestore_layer("test", project=firestore_emulator)
    idx_user, data_user = _write_some_data(layer)
    data = layer[idx_user]
    assert data == data_user
    assert layer[("key0", "col1")] == "val0"


def test_query_and_keys(firestore_emulator) -> None:
    layer = build_firestore_layer("test", project=firestore_emulator)
    _write_some_data(layer)
    col_filter0 = {"col1": ["val0"]}
    result = layer.query(column_filter=col_filter0)
    assert "key0" in result and len(result) == 1

    col_filter1 = {"col1": ["val1"]}
    result = layer.query(column_filter=col_filter1)
    assert "key2" in result and len(result) == 1

    col_filter2 = {"-tags": ["tag1"]}
    result = layer.query(column_filter=col_filter2)
    assert "key1" in result and "key2" in result and len(result) == 2


def test_inequality_filters(firestore_emulator) -> None:
    layer = build_firestore_layer("test", project=firestore_emulator)
    _write_some_query_data(layer)
    col_filter = {">col2": [2]}
    result = layer.query(column_filter=col_filter)
    assert "key3" in result and "key4" in result and len(result) == 2
    layer.clear()


def test_delete_and_clear(firestore_emulator) -> None:
    layer = build_firestore_layer("test", project=firestore_emulator)
    _write_some_data(layer)
    assert len(layer.query()) == 3

    del layer["key0"]
    assert len(layer.query()) == 2

    layer.clear()
    assert len(layer.query()) == 0


def _test_batches(layer: DBLayer, batch_size: int):
    ROW_COUNT = len(layer)
    batches = []
    batch_count = int(math.ceil(len(layer) / batch_size))
    for i in range(batch_count):
        batches.append(layer.get_batch(i, batch_size))

    batch_0 = layer.get_batch(0, batch_size)
    assert len(batches[0]) == len(batch_0)

    batch_keys = [k for b in batches for k in b.keys()]
    batch_sizes = [len(b) for b in batches]

    assert sum(batch_sizes) == ROW_COUNT
    assert len(batch_keys) == ROW_COUNT
    assert len(set(batch_keys)) == ROW_COUNT

    # test batch_size > len(layer), must return all rows
    batch = layer.get_batch(0, ROW_COUNT + 1)
    assert len(batch) == ROW_COUNT

    # test out of bounds error
    with pytest.raises(IndexError):
        layer.get_batch(batch_count, batch_size)


def test_batching(firestore_emulator, mocker) -> None:
    layer = build_firestore_layer("test", project=firestore_emulator)

    ROW_COUNT = 150
    COLS = ("col_a", "col_b")
    mocker.patch(
        "zetta_utils.layer.db_layer.firestore.backend.FirestoreBackend.__len__",
        return_value=ROW_COUNT,
    )
    rows: DBDataT = [
        dict(zip(COLS, [randint(0, 1000), randint(0, 1000)])) for _ in range(1, ROW_COUNT + 1)
    ]
    row_keys = list(str(x) for x in range(1, ROW_COUNT + 1))  # cannot use 0 as key in Datastore.
    layer[(row_keys, COLS)] = rows

    assert len(layer) == ROW_COUNT

    _test_batches(layer, 75)  # test even batch size, total divisible by batch
    _test_batches(layer, 45)  # test odd batch size


def test_with_changes(firestore_emulator) -> None:
    backend = FirestoreBackend(firestore_emulator, firestore_emulator, project=firestore_emulator)
    backend2 = backend.with_changes(collection=backend.collection, project=backend.project)
    assert isinstance(backend2, FirestoreBackend)

    with pytest.raises(KeyError):
        backend2 = backend.with_changes(
            collection=backend.collection,
            project=backend.project,
            some_key="test",
        )


def test_pickle(firestore_emulator) -> None:
    layer = build_firestore_layer("test", project=firestore_emulator)
    layer_backend = cast(FirestoreBackend, layer.backend)
    assert layer_backend.client is not None  # ensure client is initialized

    layer2 = pickle.loads(pickle.dumps(layer))
    layer_backend2 = cast(FirestoreBackend, layer2.backend)
    assert layer_backend2.project == layer_backend.project
    assert layer_backend2.collection == layer_backend.collection
    assert layer_backend2.name == layer_backend.name
