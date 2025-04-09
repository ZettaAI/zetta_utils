# pylint: disable=unused-argument,redefined-outer-name

from typing import cast

import pytest

from zetta_utils.db_annotations import annotation, collection, layer, layer_group


@pytest.fixture
def collection_and_layer_group():
    user = "john_doe"
    collection_id = collection.add_collection("collection_x0", user, "this is a test")
    layer_groups = layer_group.read_layer_groups(collection_ids=[collection_id])
    annotations = annotation.read_annotations(collection_ids=[collection_id])
    layer_group.delete_layer_groups(list(layer_groups.keys()))
    annotation.delete_annotations(list(annotations.keys()))

    layer_id0 = layer.add_layer("test_layer0", "precomputed://test0", "this is a test")
    layer_id1 = layer.add_layer("test_layer1", "precomputed://test1", "this is a test")

    layer_group_id = layer_group.add_layer_group(
        name="layer_group_x0",
        collection_id=collection_id,
        user=user,
        layers=[layer_id0, layer_id1],
        comment="this is a test",
    )
    yield (collection_id, layer_group_id)
    layer_group.delete_layer_group(layer_group_id)
    layer_groups = layer_group.read_layer_groups(collection_ids=[collection_id])
    annotations = annotation.read_annotations(collection_ids=[collection_id])
    layer_group.delete_layer_groups(list(layer_groups.keys()))
    annotation.delete_annotations(list(annotations.keys()))
    collection.delete_collection(collection_id=collection_id)


def test_add_update_delete_annotation(
    firestore_emulator,
    annotations_db,
    collections_db,
    layer_groups_db,
    layers_db,
    collection_and_layer_group,
):
    collection_id, layer_group_id = collection_and_layer_group
    annotation_raw = {
        "pointA": [1, 1, 1],
        "pointB": [1, 1, 5],
        "type": "line",
        "id": "9dd7fcd729915fcb0e32a2d92db0ca1fe5a82f02",
    }
    _id = annotation.add_annotation(
        annotation.parse_ng_annotations([annotation_raw])[0],
        collection_id=collection_id,
        layer_group_id=layer_group_id,
        comment="this is a test",
        tags=["tag0", "tag1"],
    )

    _annotation = annotation.read_annotation(_id)
    assert _annotation.collection == collection_id
    assert _annotation.layer_group == layer_group_id
    assert len(cast(list, _annotation.tags)) == 2
    assert len(_annotation.selected_segments) == 0

    _annotation_d = _annotation.to_dict()
    assert _annotation_d["collection"] == _annotation.collection

    annotation.update_annotation(
        _id,
        collection_id=collection_id,
        layer_group_id=layer_group_id,
        comment="this is also a test",
        tags=["tag2"],
        selected_segments={"yo": [1, 2, 3]},
    )
    _annotation = annotation.read_annotation(_id)
    assert len(cast(list, _annotation.tags)) == 1
    assert cast(list, _annotation.tags)[0] == "tag2"
    assert _annotation.selected_segments == {"yo": [1, 2, 3]}

    annotation.delete_annotation(_id)
    with pytest.raises(KeyError):
        annotation.read_annotation(_id)


def test_add_update_annotations(
    firestore_emulator,
    annotations_db,
    collections_db,
    layer_groups_db,
    layers_db,
    collection_and_layer_group,
):
    collection_id, layer_group_id = collection_and_layer_group
    annotations_raw = [
        {
            "pointA": [1, 1, 1],
            "pointB": [1, 1, 5],
            "type": "line",
            "id": "9dd7fcd729915fcb0e32a2d92db0ca1fe5a82f02",
        },
        {
            "point": [1, 1, 2],
            "type": "point",
            "id": "f8e5c028b6d7fddcdcd67242d9e97b9afad10078",
        },
        {
            "pointA": [28556.974609375, 13762.66796875, 61.5],
            "pointB": [28617.99609375, 13811.0126953125, 62.5],
            "type": "axis_aligned_bounding_box",
            "id": "6fdfd685cc440a6106a089113869f5043cb18c2c",
        },
        {
            "center": [28696.703125, 13757.951171875, 61.5],
            "radii": [28.595703125, 25.3515625, 0],
            "type": "ellipsoid",
            "id": "3d3d9cab641f9d1b5c81cd6dfe40999891999a59",
        },
    ]
    _ids = annotation.add_annotations(
        annotation.parse_ng_annotations(annotations_raw),
        collection_id=collection_id,
        layer_group_id=layer_group_id,
        comment="this is a test",
        tags=["tag0", "tag1"],
        selected_segments=[{"yo": ["0", "1"]} for _ in annotations_raw],
    )

    _annotations = annotation.read_annotations(annotation_ids=_ids)
    assert _annotations[0].ng_annotation.type == "line"
    assert _annotations[1].ng_annotation.type == "point"
    assert len(_annotations[0].tags) == 2
    assert len(_annotations[1].tags) == 2
    assert _annotations[0].selected_segments == {"yo": ["0", "1"]}

    annotation.update_annotations(
        _ids,
        collection_id=collection_id,
        layer_group_id=layer_group_id,
        comment="this is also a test",
        tags=["tag2"],
        selected_segments=[{} for _ in _ids],
    )
    _annotations = annotation.read_annotations(annotation_ids=_ids)
    assert len(_annotations[0].tags) == 1
    assert len(_annotations[1].tags) == 1
    assert _annotations[0].selected_segments == {}


def test_read_delete_annotations(firestore_emulator, annotations_db, collection_and_layer_group):
    collection_id, layer_group_id = collection_and_layer_group
    ng_annotations = annotation.parse_ng_annotations(
        [
            {
                "pointA": [1, 1, 1],
                "pointB": [1, 1, 5],
                "type": "line",
                "id": "9dd7fcd729915fcb0e32a2d92db0ca1fe5a82f02",
            },
            {
                "point": [1, 1, 2],
                "type": "point",
                "id": "f8e5c028b6d7fddcdcd67242d9e97b9afad10078",
            },
        ]
    )

    _id0 = annotation.add_annotation(
        ng_annotations[0],
        collection_id=collection_id,
        layer_group_id=layer_group_id,
        comment="this is a test",
        tags=["tag0"],
    )

    _id1 = annotation.add_annotation(
        ng_annotations[1],
        collection_id=collection_id,
        layer_group_id=layer_group_id,
        comment="this is a test",
        tags=["tag1"],
    )

    _annotations = annotation.read_annotations(collection_ids=[collection_id])
    assert len(_annotations) == 2

    _annotations = annotation.read_annotations(layer_group_ids=[layer_group_id])
    assert len(_annotations) == 2

    _annotations = annotation.read_annotations(tags=["tag0"])
    assert len(_annotations) == 1

    annotation.delete_annotations([_id0, _id1])

    with pytest.raises(KeyError):
        annotation.read_annotation(_id0)

    with pytest.raises(KeyError):
        annotation.read_annotation(_id1)
