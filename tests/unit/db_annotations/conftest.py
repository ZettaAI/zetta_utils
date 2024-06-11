# pylint: disable=invalid-name

import pytest

from zetta_utils.db_annotations import annotation, collection, layer, layer_group
from zetta_utils.layer.db_layer.firestore import build_firestore_layer


@pytest.fixture(scope="session")
def annotations_db(firestore_emulator):
    db = build_firestore_layer(annotation.DB_NAME, project=firestore_emulator)
    annotation.ANNOTATIONS_DB = db
    return annotation.ANNOTATIONS_DB


@pytest.fixture(scope="session")
def collections_db(firestore_emulator):
    db = build_firestore_layer(collection.DB_NAME, project=firestore_emulator)
    collection.COLLECTIONS_DB = db
    return collection.COLLECTIONS_DB


@pytest.fixture(scope="session")
def layer_groups_db(firestore_emulator):
    db = build_firestore_layer(layer_group.DB_NAME, project=firestore_emulator)
    layer_group.LAYER_GROUPS_DB = db
    return layer_group.LAYER_GROUPS_DB


@pytest.fixture(scope="session")
def layers_db(firestore_emulator):
    db = build_firestore_layer(layer.DB_NAME, project=firestore_emulator)
    layer.LAYERS_DB = db
    return layer.LAYERS_DB
