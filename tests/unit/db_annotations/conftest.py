# pylint: disable=invalid-name

from typing import cast

import pytest

from zetta_utils.db_annotations import annotation, collection, layer, layer_group
from zetta_utils.layer.db_layer.datastore import DatastoreBackend, build_datastore_layer


@pytest.fixture(scope="session")
def annotations_db(datastore_emulator):
    db = build_datastore_layer(annotation.DB_NAME, datastore_emulator)
    cast(DatastoreBackend, db.backend).exclude_from_indexes = annotation.NON_INDEXED_COLS
    annotation.ANNOTATIONS_DB = db
    return annotation.ANNOTATIONS_DB


@pytest.fixture(scope="session")
def collections_db(datastore_emulator):
    db = build_datastore_layer(collection.DB_NAME, datastore_emulator)
    cast(DatastoreBackend, db.backend).exclude_from_indexes = collection.NON_INDEXED_COLS
    collection.COLLECTIONS_DB = db
    return collection.COLLECTIONS_DB


@pytest.fixture(scope="session")
def layer_groups_db(datastore_emulator):
    db = build_datastore_layer(layer_group.DB_NAME, datastore_emulator)
    cast(DatastoreBackend, db.backend).exclude_from_indexes = layer_group.NON_INDEXED_COLS
    layer_group.LAYER_GROUPS_DB = db
    return layer_group.LAYER_GROUPS_DB


@pytest.fixture(scope="session")
def layers_db(datastore_emulator):
    db = build_datastore_layer(layer.DB_NAME, datastore_emulator)
    cast(DatastoreBackend, db.backend).exclude_from_indexes = layer.NON_INDEXED_COLS
    layer.LAYERS_DB = db
    return layer.LAYERS_DB
