from pprint import pprint

from neuroglancer.viewer_state import (
    AxisAlignedBoundingBoxAnnotation,
    EllipsoidAnnotation,
    LineAnnotation,
    PointAnnotation,
)

from .annotation import add_annotations, read_annotations
from .collection import add_collection, read_collections
from .layer import add_layer, read_layers
from .layer_group import add_layer_group, read_layer_groups


def test_add_layers():
    _id1 = add_layer(
        "src", "precomputed://gs://dkronauer-ant-001-alignment/production-240118/i.0_lr0.001"
    )
    _id2 = add_layer("tgt", "precomputed://gs://dkronauer-ant-001-raw/brain")
    layer_ids = [_id1, _id2]
    pprint(read_layers(layer_ids))
    return layer_ids


def test_add_collections():
    _id = add_collection("test_c1", "akhilesh", "collection test")
    pprint(read_collections([_id]))
    return _id


def test_add_layer_groups(layer_ids):
    _id = add_layer_group("test_lg1", "akhilesh", layer_ids, "layer group test")
    pprint(read_layer_groups([_id]))
    return _id


def test_add_annotations(collection_ids, layer_group_id):
    annotations_raw = [
        {
            "pointA": [28695.8671875, 13933.6025390625, 61.5],
            "pointB": [28672.55859375, 13908.0263671875, 61.5],
            "type": "line",
            "id": "9dd7fcd729915fcb0e32a2d92db0ca1fe5a82f02",
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
        {
            "point": [28791.919921875, 13775.048828125, 61.5],
            "type": "point",
            "id": "f8e5c028b6d7fddcdcd67242d9e97b9afad10078",
        },
        {
            "point": [28769.515625, 13811.6025390625, 61.5],
            "type": "point",
            "id": "d11db2361eef5f37d03b6569a768e9165c3a9006",
        },
        {
            "pointA": [28469.126953125, 13868.2021484375, 61.5],
            "pointB": [28487.994140625, 13835.185546875, 61.5],
            "type": "line",
            "id": "106f635eaec8c88764f3550e09e587019101345f",
        },
    ]

    annotations = []
    annotation_ids = []
    for ann in annotations_raw:
        _type = ann.pop("type")
        if _type == "line":
            annotations.append(LineAnnotation(**ann))
        elif _type == "ellipsoid":
            annotations.append(EllipsoidAnnotation(**ann))
        elif _type == "point":
            annotations.append(PointAnnotation(**ann))
        else:
            annotations.append(AxisAlignedBoundingBoxAnnotation(**ann))
        annotation_ids.append(ann["id"])

    add_annotations(annotations[:4], collection_ids[0], layer_group_id, tags=["tag0", "tag1"])
    add_annotations(annotations[4:], collection_ids[1], layer_group_id, tags=["tag2"])
    # result = read_annotations(annotation_ids=annotation_ids)
    # for _id, r in zip(annotation_ids, result):
    #     print(_id)
    #     pprint(r)
    #     print()


if __name__ == "__main__":
    # layer_ids = test_add_layers()
    # print()
    # collection_id = test_add_collections()
    # print()
    # layer_group_id = test_add_layer_groups(layer_ids)
    # print()
    _collection_ids = [
        "259c4d57-30e6-44ae-8f05-3cf981d07cda",
        "ca03d4ea-cf58-4454-b7ce-9735188e9b03",
    ]
    # test_add_annotations(_collection_ids, "5fa19d71-6741-49e2-b6fc-aa2fca508e20")
    # print()
    pprint(read_annotations(collection_ids=_collection_ids[1:]))
    print()
    pprint(read_annotations(tags=["tag2"]))
    print()
    pprint(read_annotations(tags=["tag0", "tag2"]))
