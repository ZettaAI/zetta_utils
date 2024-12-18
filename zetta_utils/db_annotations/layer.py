"""Manage layers in a DB Layer."""

from __future__ import annotations

import uuid
from typing import List, Mapping, cast, overload

import attrs

from zetta_utils.layer.db_layer import DBRowDataT
from zetta_utils.layer.db_layer.firestore import build_firestore_layer

from . import constants

DB_NAME = "layers"
INDEXED_COLS = ("name", "source")
NON_INDEXED_COLS = ("comment",)

LAYERS_DB = build_firestore_layer(
    DB_NAME,
    project=constants.PROJECT,
    database=constants.DATABASE,
)


@attrs.mutable
class LayerDBEntry:
    id: str
    name: str
    source: str

    @staticmethod
    def from_dict(layer_id: str, raw_dict: Mapping) -> LayerDBEntry:
        return LayerDBEntry(
            id=layer_id,
            name=raw_dict["name"],
            source=raw_dict["source"],
        )


def read_layer(layer_id: str) -> LayerDBEntry:
    idx = (layer_id, INDEXED_COLS + NON_INDEXED_COLS)
    result_raw = LAYERS_DB[idx]
    result = LayerDBEntry.from_dict(layer_id, result_raw)
    return result


@overload
def read_layers() -> dict[str, LayerDBEntry]:
    ...


@overload
def read_layers(*, layer_ids: list[str]) -> list[LayerDBEntry]:
    ...


def read_layers(*, layer_ids=None):
    if layer_ids is None:
        result_raw = LAYERS_DB.query()
        return {k: LayerDBEntry.from_dict(k, cast(dict, v)) for k, v in result_raw.items()}
    else:
        idx = (layer_ids, INDEXED_COLS + NON_INDEXED_COLS)
        result_raw_list = cast(List, LAYERS_DB[idx])
        return [
            LayerDBEntry.from_dict(layer_ids[i], result_raw_list[i])
            for i in range(len(result_raw_list))
        ]


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


def delete_layer(layer_id: str):  # pragma: no cover
    del LAYERS_DB[layer_id]
