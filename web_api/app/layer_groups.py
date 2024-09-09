# pylint: disable=all # type: ignore
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request

from zetta_utils.db_annotations.layer_group import (
    add_layer_group,
    delete_layer_group,
    delete_layer_groups,
    read_layer_group,
    read_layer_groups,
    update_layer_group,
)

from .utils import generic_exception_handler

api = FastAPI()


@api.exception_handler(Exception)
async def generic_handler(request: Request, exc: Exception):
    return generic_exception_handler(request, exc)


@api.get("/single/{layer_group_id}")
async def read_single(layer_group_id: str):
    return read_layer_group(layer_group_id)


@api.post("/single")
async def add_single(
    name: str,
    collection_id: str,
    user: str,
    layers: list[str] | None = None,
    comment: str | None = None,
):
    try:
        return add_layer_group(
            name=name, collection_id=collection_id, user=user, layers=layers, comment=comment
        )
    except KeyError:
        raise HTTPException(
            status_code=409, detail=f"`{collection_id}` already has layer group `{name}`."
        )


@api.put("/single")
async def update_single(
    layer_group_id: str,
    user: str,
    collection_id: str | None = None,
    name: str | None = None,
    layers: list[str] | None = None,
    comment: str | None = None,
):
    update_layer_group(
        layer_group_id,
        user=user,
        collection_id=collection_id,
        name=name,
        layers=layers,
        comment=comment,
    )


@api.delete("/single")
async def delete_single(layer_group_id: str):
    delete_layer_group(layer_group_id)


@api.get("/multiple")
async def read_multiple(
    layer_group_ids: Annotated[list[str] | None, Query()] = None,
    collection_ids: Annotated[list[str] | None, Query()] = None,
):
    if layer_group_ids:
        return read_layer_groups(layer_group_ids=layer_group_ids)
    if collection_ids:
        layer_groups = read_layer_groups(collection_ids=collection_ids)
    else:
        layer_groups = read_layer_groups()
    response = []
    for _id, layer_group in layer_groups.items():
        layer_group.id = _id
        response.append(layer_group)
    return response


@api.delete("/multiple")
async def delete_multiple(layer_group_ids: Annotated[list[str], Query()]):
    delete_layer_groups(layer_group_ids)
