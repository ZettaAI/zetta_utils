"""Manage annotation collections in a DB Layer."""

from __future__ import annotations

import time
from typing import List, Mapping, cast, overload

import attrs

from zetta_utils.layer.db_layer import DBRowDataT
from zetta_utils.layer.db_layer.firestore import build_firestore_layer

from . import constants

DB_NAME = "collections"
INDEXED_COLS = ("name", "name_lowercase", "created_by", "created_at", "modified_by", "modified_at")
NON_INDEXED_COLS = ("comment",)

COLLECTIONS_DB = build_firestore_layer(
    DB_NAME,
    project=constants.PROJECT,
    database=constants.DATABASE,
)


@attrs.mutable
class CollectionDBEntry:
    id: str
    name: str
    created_by: str
    created_at: float
    name_lowercase: str | None
    modified_by: str | None
    modified_at: float | None
    comment: str | None

    @staticmethod
    def from_dict(collection_id: str, raw_dict: Mapping) -> CollectionDBEntry:
        return CollectionDBEntry(
            id=collection_id,
            name=raw_dict["name"],
            created_by=raw_dict["created_by"],
            created_at=raw_dict["created_at"],
            name_lowercase=raw_dict.get("name_lowercase"),
            modified_by=raw_dict.get("modified_by"),
            modified_at=raw_dict.get("modified_at"),
            comment=raw_dict.get("comment"),
        )


def read_collection(collection_id: str) -> CollectionDBEntry:
    idx = (collection_id, INDEXED_COLS + NON_INDEXED_COLS)
    result_raw = COLLECTIONS_DB[idx]
    result = CollectionDBEntry.from_dict(collection_id, result_raw)
    return result


@overload
def read_collections() -> dict[str, CollectionDBEntry]:
    ...


@overload
def read_collections(*, collection_ids: list[str]) -> list[CollectionDBEntry]:
    ...


def read_collections(*, collection_ids=None):
    if collection_ids is None:
        result_raw = COLLECTIONS_DB.query()
        return {k: CollectionDBEntry.from_dict(k, cast(dict, v)) for k, v in result_raw.items()}
    else:
        idx = (collection_ids, INDEXED_COLS + NON_INDEXED_COLS)
        result_raw_list = cast(List, COLLECTIONS_DB[idx])
        return [
            CollectionDBEntry.from_dict(collection_ids[i], result_raw_list[i])
            for i in range(len(result_raw_list))
        ]


def add_collection(name: str, user: str, comment: str | None = None) -> str:
    collection_id = name
    if collection_id in COLLECTIONS_DB:
        raise KeyError(f"{collection_id} already exists.")
    col_keys = INDEXED_COLS + NON_INDEXED_COLS
    row: DBRowDataT = {
        "name": name,
        "created_by": user,
        "created_at": time.time(),
        "modified_by": user,
        "modified_at": time.time(),
    }
    if comment:
        row["comment"] = comment
    COLLECTIONS_DB[(collection_id, col_keys)] = row
    return collection_id


def update_collection(
    collection_id: str, user: str, name: str | None = None, comment: str | None = None
):
    col_keys = INDEXED_COLS + NON_INDEXED_COLS
    row: DBRowDataT = {"modified_by": user, "modified_at": time.time()}
    if name:
        row["name"] = name
    if comment:
        row["comment"] = comment
    COLLECTIONS_DB[(collection_id, col_keys)] = row


def delete_collection(collection_id: str):
    del COLLECTIONS_DB[collection_id]


def delete_collections(collection_ids: list[str]):
    del COLLECTIONS_DB[collection_ids]
