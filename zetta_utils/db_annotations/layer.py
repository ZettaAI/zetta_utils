"""Manage layers in a DB Layer."""

from __future__ import annotations

import uuid

from zetta_utils.layer.db_layer import DBRowDataT
from zetta_utils.layer.db_layer.datastore import build_datastore_layer

from . import constants

DB_NAME = "layers"
INDEXED_COLS = ("name", "source")
NON_INDEXED_COLS = ("comment",)

LAYERS_DB = build_datastore_layer(
    DB_NAME,
    project=constants.PROJECT,
    database=constants.DATABASE,
    exclude_from_indexes=NON_INDEXED_COLS,
)


def read_layer(layer_id: str) -> DBRowDataT:
    idx = (layer_id, INDEXED_COLS + NON_INDEXED_COLS)
    return LAYERS_DB[idx]


def read_layers(layer_ids: list[str]) -> list[DBRowDataT]:
    idx = (layer_ids, INDEXED_COLS + NON_INDEXED_COLS)
    return LAYERS_DB[idx]


def add_layer(name: str, source: str, comment: str | None = None) -> str:
    layer_id = str(uuid.uuid4())
    col_keys = INDEXED_COLS + NON_INDEXED_COLS
    row: DBRowDataT = {"name": name, "source": source}
    if comment:
        row["comment"] = comment
    LAYERS_DB[(layer_id, col_keys)] = row
    return layer_id


def update_layer(
    layer_id: str,
    *,
    name: str | None = None,
    source: str | None = None,
    comment: str | None = None,
):
    col_keys = INDEXED_COLS + NON_INDEXED_COLS
    row: DBRowDataT = {}
    if name:
        row["name"] = name
    if comment:
        row["comment"] = comment
    if source:
        row["source"] = source
    LAYERS_DB[(layer_id, col_keys)] = row


def delete_layer(layer_id: str):
    raise NotImplementedError()
