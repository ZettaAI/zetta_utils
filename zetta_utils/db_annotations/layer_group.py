"""Manage layer groups in a DB Layer."""

from __future__ import annotations

from typing import List, Mapping, cast, overload

import attrs

from zetta_utils.layer.db_layer import DBRowDataT
from zetta_utils.layer.db_layer.firestore import build_firestore_layer

from . import constants

DB_NAME = "layer_groups"
INDEXED_COLS = ("name", "layers", "collection", "created_by", "modified_by")
NON_INDEXED_COLS = ("comment",)

LAYER_GROUPS_DB = build_firestore_layer(
    DB_NAME,
    project=constants.PROJECT,
    database=constants.DATABASE,
)


@attrs.mutable
class LayerGroupDBEntry:
    id: str
    name: str
    collection: str
    layers: list[str]
    created_by: str
    modified_by: str | None

    @staticmethod
    def from_dict(layer_group_id: str, raw_dict: Mapping) -> LayerGroupDBEntry:
        return LayerGroupDBEntry(
            id=layer_group_id,
            name=raw_dict["name"],
            created_by=raw_dict["created_by"],
            modified_by=raw_dict.get("modified_by"),
            collection=raw_dict["collection"],
            layers=raw_dict.get("layers", []),
        )


def read_layer_group(layer_group_id: str) -> LayerGroupDBEntry:
    idx = (layer_group_id, INDEXED_COLS + NON_INDEXED_COLS)
    result_raw = LAYER_GROUPS_DB[idx]
    result = LayerGroupDBEntry.from_dict(layer_group_id=layer_group_id, raw_dict=result_raw)
    return result


@overload
def read_layer_groups(*, layer_group_ids: list[str]) -> list[LayerGroupDBEntry]:
    ...


@overload
def read_layer_groups(*, collection_ids: list[str] | None = None) -> dict[str, LayerGroupDBEntry]:
    ...


def read_layer_groups(*, layer_group_ids=None, collection_ids=None):
    _filter = {}
    if collection_ids:
        _filter["collection"] = collection_ids
        result_raw = LAYER_GROUPS_DB.query(column_filter=_filter)
        return {k: LayerGroupDBEntry.from_dict(k, cast(dict, v)) for k, v in result_raw.items()}
    elif layer_group_ids is None:
        result_raw = LAYER_GROUPS_DB.query()
        return {k: LayerGroupDBEntry.from_dict(k, cast(dict, v)) for k, v in result_raw.items()}
    else:
        idx = (layer_group_ids, INDEXED_COLS + NON_INDEXED_COLS)
        result_raw_list = cast(List, LAYER_GROUPS_DB[idx])
        return [
            LayerGroupDBEntry.from_dict(layer_group_ids[i], result_raw_list[i])
            for i in range(len(result_raw_list))
        ]


def add_layer_group(
    *,
    name: str,
    collection_id: str,
    user: str,
    layers: list[str] | None = None,
    comment: str | None = None,
) -> str:
    layer_group_id = f"{collection_id}:{name}"
    if layer_group_id in LAYER_GROUPS_DB:
        raise KeyError(f"{layer_group_id} already exists.")
    col_keys = INDEXED_COLS + NON_INDEXED_COLS
    row: DBRowDataT = {"name": name, "collection": collection_id, "created_by": user}
    if layers:
        row["layers"] = list(set(layers))
    if comment:
        row["comment"] = comment
    LAYER_GROUPS_DB[(layer_group_id, col_keys)] = row
    return layer_group_id


def update_layer_group(
    layer_group_id: str,
    *,
    user: str,
    collection_id: str | None = None,
    name: str | None = None,
    layers: list[str] | None = None,
    comment: str | None = None,
):
    col_keys = INDEXED_COLS + NON_INDEXED_COLS
    row: DBRowDataT = {"modified_by": user}
    if collection_id:
        row["collection"] = collection_id
    if name:
        row["name"] = name
    if layers:
        row["layers"] = list(set(layers))
    if comment:
        row["comment"] = comment
    LAYER_GROUPS_DB[(layer_group_id, col_keys)] = row


def delete_layer_group(layer_group_id: str):
    del LAYER_GROUPS_DB[layer_group_id]


def delete_layer_groups(layer_group_ids: list[str]):
    del LAYER_GROUPS_DB[layer_group_ids]
