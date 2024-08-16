from typing import Annotated

from fastapi import FastAPI, Query

from zetta_utils.db_annotations.collection import (
    add_collection,
    read_collection,
    read_collections,
    update_collection,
)

api = FastAPI()


@api.get("/single/{collection_id}")
async def read_single(collection_id: str):
    return read_collection(collection_id)


@api.post("/single")
async def add_single(name: str, user: str, comment: str | None = None):
    return add_collection(name, user, comment=comment)


@api.put("/single")
async def update_single(
    collection_id: str, user: str, name: str | None = None, comment: str | None = None
):
    update_collection(collection_id, user, name=name, comment=comment)


@api.get("/multiple")
async def read_multiple(collection_ids: Annotated[list[str] | None, Query()] = None):
    return read_collections(collection_ids)
