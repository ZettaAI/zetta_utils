# pylint: disable=unused-argument

from zetta_utils.db_annotations import layer


def test_add_update_layer(firestore_emulator, layers_db):
    old_name = "test_layer0"
    _id = layer.add_layer(old_name, "precomputed://test", comment="this is a test")
    _layer = layer.read_layer(_id)
    assert _layer["name"] == old_name
    assert _layer["source"] == "precomputed://test"

    layer.update_layer(
        _id, name="test_layer1", source="precomputed://test", comment="this is a test"
    )
    _layer = layer.read_layer(_id)
    assert _layer["name"] != old_name


def test_read_layers(firestore_emulator, layers_db):
    layer_id0 = layer.add_layer("test_layer0", "precomputed://test0", "this is a test")
    layer_id1 = layer.add_layer("test_layer1", "precomputed://test1", "this is a test")
    _layers = layer.read_layers(layer_ids=[layer_id0, layer_id1])

    assert _layers[0]["name"] == "test_layer0"
    assert _layers[0]["source"] == "precomputed://test0"
    assert _layers[1]["name"] == "test_layer1"
    assert _layers[1]["source"] == "precomputed://test1"
