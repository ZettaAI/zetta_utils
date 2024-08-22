# pylint: disable=unused-argument

from typing import cast

import pytest

from zetta_utils.db_annotations import collection, layer, layer_group


def test_add_update_delete_layer_group(firestore_emulator, layer_groups_db):
    user = "john_doe"
    collection_name = "test_collection0"
    collection_id = collection.add_collection(collection_name, user, "this is a test")

    layer_id0 = layer.add_layer("test_layer0", "precomputed://test0", "this is a test")
    layer_id1 = layer.add_layer("test_layer1", "precomputed://test1", "this is a test")
    layers_ids = [layer_id0, layer_id1]

    old_name = "test_layer_group0"
    _id = layer_group.add_layer_group(
        name=old_name,
        collection_id=collection_id,
        user=user,
        layers=layers_ids,
        comment="this is a test",
    )
    _layer_group = layer_group.read_layer_group(_id)
    assert _layer_group["name"] == old_name
    assert len(cast(list, _layer_group["layers"])) == 2

    layer_group.update_layer_group(
        _id,
        name="test_layerG1",
        collection_id=collection_id,
        user=user,
        layers=layers_ids,
        comment="this is a test",
    )
    _layer = layer_group.read_layer_group(_id)
    assert _layer["name"] != old_name

    layer_group.delete_layer_group(_id)
    with pytest.raises(KeyError):
        layer_group.read_layer_group(_id)


def test_read_delete_layer_groups(firestore_emulator, layer_groups_db):
    user = "john_doe"
    collection_name = "test_collection0"
    collection_id = collection.add_collection(collection_name, user, "this is a test")

    layer_id0 = layer.add_layer("test_layer0", "precomputed://test0", "this is a test")
    layer_id1 = layer.add_layer("test_layer1", "precomputed://test1", "this is a test")
    layers_ids = [layer_id0, layer_id1]

    _id0 = layer_group.add_layer_group(
        name="test_layer_group0",
        collection_id=collection_id,
        user=user,
        layers=layers_ids,
        comment="this is a test",
    )

    _id1 = layer_group.add_layer_group(
        name="test_layer_group1",
        collection_id=collection_id,
        user=user,
        layers=layers_ids,
        comment="this is a test",
    )

    _layer_groups = layer_group.read_layer_groups(layer_group_ids=[_id0, _id1])
    assert _layer_groups[0]["name"] == "test_layer_group0"
    assert len(cast(list, _layer_groups[0]["layers"])) == len(
        cast(list, _layer_groups[1]["layers"])
    )

    _layer_groups1 = layer_group.read_layer_groups()
    assert len(_layer_groups1) == 2

    _layer_groups2 = layer_group.read_layer_groups(collection_ids=[collection_id])
    assert len(_layer_groups2) == 2

    layer_group.delete_layer_groups([_id0, _id1])

    with pytest.raises(KeyError):
        layer_group.read_layer_group(_id0)

    with pytest.raises(KeyError):
        layer_group.read_layer_group(_id1)
