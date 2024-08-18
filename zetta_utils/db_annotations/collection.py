"""Manage annotation collections in a DB Layer."""

from __future__ import annotations

import time
import uuid
from typing import overload

from zetta_utils.layer.db_layer import DBRowDataT
from zetta_utils.layer.db_layer.firestore import build_firestore_layer

from . import constants

DB_NAME = "collections"
INDEXED_COLS = ("name", "created_by", "created_at", "modified_by", "modified_at")
NON_INDEXED_COLS = ("comment",)

COLLECTIONS_DB = build_firestore_layer(
    DB_NAME,
    project=constants.PROJECT,
    database=constants.DATABASE,
)


def read_collection(collection_id: str) -> DBRowDataT:
    idx = (collection_id, INDEXED_COLS + NON_INDEXED_COLS)
    return COLLECTIONS_DB[idx]


@overload
def read_collections() -> dict[str, DBRowDataT]:
    ...


@overload
def read_collections(*, collection_ids: list[str]) -> list[DBRowDataT]:
    ...


def read_collections(
    *, collection_ids: list[str] | None = None
) -> list[DBRowDataT] | dict[str, DBRowDataT]:
    if collection_ids is None:
        return COLLECTIONS_DB.query()
    idx = (collection_ids, INDEXED_COLS + NON_INDEXED_COLS)
    return COLLECTIONS_DB[idx]


def add_collection(name: str, user: str, comment: str | None = None) -> str:
    collection_id = str(uuid.uuid4())
    col_keys = INDEXED_COLS + NON_INDEXED_COLS
    row: DBRowDataT = {"name": name, "created_by": user, "created_at": time.time()}
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
