# pylint: disable=all # type: ignore
from typing import Annotated

from attrs import asdict
from fastapi import FastAPI, HTTPException, Query, Request

from zetta_utils.db_annotations.collection import (
    add_collection,
    delete_collection,
    delete_collections,
    read_collection,
    read_collections,
    update_collection,
)
from zetta_utils.db_annotations.deletion import cascade_delete_collections
from zetta_utils.db_annotations.layer_group import (
    delete_layer_groups,
    read_layer_groups,
)

from .utils import generic_exception_handler

api = FastAPI()


@api.exception_handler(Exception)
async def generic_handler(request: Request, exc: Exception):
    return generic_exception_handler(request, exc)


@api.get("/single/{collection_id}")
async def read_single(collection_id: str):
    return asdict(read_collection(collection_id))


@api.post("/single")
async def add_single(name: str, user: str, comment: str | None = None):
    try:
        return add_collection(name, user, comment=comment)
    except KeyError:
        raise HTTPException(status_code=409, detail=f"`{name}` exists.")


@api.put("/single")
async def update_single(
    collection_id: str, user: str, name: str | None = None, comment: str | None = None
):
    update_collection(collection_id, user, name=name, comment=comment)


@api.delete("/single")
async def delete_single(collection_id: str, cascade: Annotated[bool, Query()] = True):
    if cascade:
        cascade_delete_collections([collection_id])
    delete_collection(collection_id)


@api.get("/multiple")
async def read_multiple(collection_ids: Annotated[list[str] | None, Query()] = None):
    if collection_ids:
        return read_collections(collection_ids=collection_ids)
    collections = read_collections()
    response = []
    for _id, collection in collections.items():
        collection.id = _id
        response.append(asdict(collection))
    return response


@api.delete("/multiple")
async def delete_multiple(
    collection_ids: Annotated[list[str], Query()], cascade: Annotated[bool, Query()] = True
):
    if cascade:
        cascade_delete_collections(collection_ids)
    else:
        delete_collections(collection_ids)
