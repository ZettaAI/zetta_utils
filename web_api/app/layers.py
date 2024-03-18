from fastapi import FastAPI

from zetta_utils.db_annotations.layer import (
    add_layer,
    read_layer,
    read_layers,
    update_layer,
)

api = FastAPI()


@api.get("/single/{layer_id}")
async def read_single(layer_id: str):
    return read_layer(layer_id)


@api.post("/single")
async def add_single(name: str, source: str, comment: str | None = None):
    return add_layer(name, source, comment=comment)


@api.put("/single")
async def update_single(
    layer_id: str, name: str | None = None, source: str | None = None, comment: str | None = None
):
    update_layer(layer_id, name=name, source=source, comment=comment)


@api.get("/multiple")
async def read_multiple(layer_ids: list[str]):
    return read_layers(layer_ids)
