# pylint: disable=unused-argument

import pytest

from zetta_utils.db_annotations import annotation, collection, layer_group


@pytest.fixture
def collection_x0():
    yield from collection_maker("collection_x0")


@pytest.fixture
def collection_x1():
    yield from collection_maker("collection_x1")


def collection_maker(collection_name: str):
    user = "john_doe"
    collection_id = collection.add_collection(collection_name, user, "this is a test")
    layer_groups = layer_group.read_layer_groups(collection_ids=[collection_id])
    annotations = annotation.read_annotations(collection_ids=[collection_id])
    layer_group.delete_layer_groups(list(layer_groups.keys()))
    annotation.delete_annotations(list(annotations.keys()))

    yield collection_id

    layer_groups = layer_group.read_layer_groups(collection_ids=[collection_id])
    annotations = annotation.read_annotations(collection_ids=[collection_id])
    layer_group.delete_layer_groups(list(layer_groups.keys()))
    annotation.delete_annotations(list(annotations.keys()))
    collection.delete_collection(collection_id=collection_id)


def test_add_update_delete_collection(firestore_emulator, collections_db):
    old_user = "john_doe"
    old_name = "test_collection0"
    _id = collection.add_collection(old_name, old_user, "this is a test")

    with pytest.raises(KeyError):
        collection.add_collection(old_name, old_user, "this is a test")

    _collection = collection.read_collection(_id)
    assert _collection.name == old_name
    assert _collection.created_by == old_user

    new_user = "jane_doe"
    collection.update_collection(
        _id, user=new_user, name="test_collection1", comment="this is also a test"
    )
    _collection = collection.read_collection(_id)
    assert _collection.name != old_name
    assert _collection.modified_by == new_user

    collection.delete_collection(_id)
    with pytest.raises(KeyError):
        collection.read_collection(_id)


def test_read_delete_collections(firestore_emulator, collections_db):
    collection_id0 = collection.add_collection("test_collection0", "john_doe", "this is a test")
    collection_id1 = collection.add_collection("test_collection1", "jane_doe", "this is a test")
    _collections = collection.read_collections(collection_ids=[collection_id0, collection_id1])

    assert _collections[0].name == "test_collection0"
    assert _collections[0].created_by == "john_doe"
    assert _collections[1].name == "test_collection1"
    assert _collections[1].created_by == "jane_doe"

    _collections1 = collection.read_collections()
    assert len(_collections1) == 2

    collection.delete_collections([collection_id0, collection_id1])

    with pytest.raises(KeyError):
        collection.read_collection(collection_id0)

    with pytest.raises(KeyError):
        collection.read_collection(collection_id1)
