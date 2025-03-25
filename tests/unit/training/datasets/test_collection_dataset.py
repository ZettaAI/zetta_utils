# pylint: disable=unused-argument,redefined-outer-name

import os
import pathlib

import pytest

from zetta_utils.db_annotations import annotation, collection, layer, layer_group
from zetta_utils.training.datasets.collection_dataset import build_collection_dataset

THIS_DIR = pathlib.Path(__file__).parent.resolve()
INFOS_DIR = THIS_DIR / "../../assets/infos/"
LAYER_X1_PATH = "file://" + os.path.join(INFOS_DIR, "layer_x1")
LAYER_X2_PATH = "file://" + os.path.join(INFOS_DIR, "layer_x2")


@pytest.fixture
def dummy_dataset_x0():
    user = "john_doe"
    collection_id = collection.add_collection("collection_x0", user, "this is a test")
    layer_groups = layer_group.read_layer_groups(collection_ids=[collection_id])
    annotations = annotation.read_annotations(collection_ids=[collection_id])
    layer_group.delete_layer_groups(list(layer_groups.keys()))
    annotation.delete_annotations(list(annotations.keys()))

    layer_id0 = layer.add_layer("layer0", LAYER_X1_PATH, "this is a test")
    layer_id1 = layer.add_layer("layer1", LAYER_X2_PATH, "this is a test")
    layer_id2 = layer.add_layer("layer2", LAYER_X2_PATH, "this is a test")

    layer_group_id = layer_group.add_layer_group(
        name="layer_group_x0",
        collection_id=collection_id,
        user=user,
        layers=[layer_id0, layer_id1, layer_id2],
        comment="this is a test",
    )

    annotation.add_annotations(
        annotation.parse_ng_annotations(
            [
                {
                    "pointA": [0, 0, 0],
                    "pointB": [128, 128, 128],
                    "type": "axis_aligned_bounding_box",
                    "id": "6fdfd685cc440a6106a089113869f5043cb18c2c",
                }
            ]
        ),
        collection_id=collection_id,
        layer_group_id=layer_group_id,
    )
    annotation.add_annotations(
        annotation.parse_ng_annotations(
            [
                {
                    "pointA": [128, 128, 128],
                    "pointB": [256, 256, 256],
                    "type": "axis_aligned_bounding_box",
                    "id": "6fdfd685cc440a6106a089113869f5043cb18c2c",
                }
            ]
        ),
        collection_id=collection_id,
        layer_group_id=layer_group_id,
    )
    yield collection_id

    layer_group.delete_layer_group(layer_group_id)
    layer_groups = layer_group.read_layer_groups(collection_ids=[collection_id])
    annotations = annotation.read_annotations(collection_ids=[collection_id])
    layer_group.delete_layer_groups(list(layer_groups.keys()))
    annotation.delete_annotations(list(annotations.keys()))
    collection.delete_collection(collection_id=collection_id)


def test_simple(
    firestore_emulator,
    annotations_db,
    collections_db,
    layer_groups_db,
    layers_db,
    dummy_dataset_x0,
):
    dset = build_collection_dataset(
        collection_name="collection_x0",
        resolution=[8, 8, 8],
        chunk_size=[1, 1, 1],
        chunk_stride=[1, 1, 1],
        layer_rename_map={"layer0": "layer00"},
        per_layer_read_procs={},
    )
    assert len(dset) == 4096 * 2
    sample = dset[0]
    assert "layer00" in sample
    assert "layer1" in sample
    assert "layer2" in sample


def test_size_exc(
    firestore_emulator,
    annotations_db,
    collections_db,
    layer_groups_db,
    layers_db,
    dummy_dataset_x0,
):
    with pytest.raises(RuntimeError):
        build_collection_dataset(
            collection_name="collection_x0",
            resolution=[8, 8, 8],
            chunk_size=[1024, 1024, 1024],
            chunk_stride=[1, 1, 1],
            layer_rename_map={"layer0": "layer00"},
        )
