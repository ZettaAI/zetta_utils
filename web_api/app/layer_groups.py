from fastapi import FastAPI

from zetta_utils.db_annotations.layer_group import (
    add_layer_group,
    read_layer_group,
    read_layer_groups,
    update_layer_group,
)

api = FastAPI()


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
    return add_layer_group(
        name=name, collection_id=collection_id, user=user, layers=layers, comment=comment
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


@api.get("/multiple")
async def read_multiple(layer_group_ids: list[str]):
    return read_layer_groups(layer_group_ids)
