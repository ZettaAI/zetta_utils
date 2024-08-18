from typing import Annotated

from fastapi import FastAPI, Query

from zetta_utils.db_annotations.collection import (
    add_collection,
    delete_collection,
    delete_collections,
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


@api.delete("/single")
async def delete_single(collection_id: str):
    delete_collection(collection_id)


@api.get("/multiple")
async def read_multiple(collection_ids: Annotated[list[str] | None, Query()] = None):
    if collection_ids:
        return read_collections(collection_ids=collection_ids)
    collections = read_collections()
    response = []
    for _id, collection in collections.items():
        collection["id"] = _id
        response.append(collection)
    return response


@api.delete("/multiple")
async def delete_multiple(collection_ids: list[str]):
    delete_collections(collection_ids)
