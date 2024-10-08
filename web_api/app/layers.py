# pylint: disable=all # type: ignore
from typing import Annotated

from attrs import asdict
from fastapi import FastAPI, Query, Request

from zetta_utils.db_annotations.layer import (
    add_layer,
    read_layer,
    read_layers,
    update_layer,
)

from .utils import generic_exception_handler

api = FastAPI()


@api.exception_handler(Exception)
async def generic_handler(request: Request, exc: Exception):
    return generic_exception_handler(request, exc)


@api.get("/single/{layer_id}")
async def read_single(layer_id: str):
    return asdict(read_layer(layer_id))


@api.post("/single")
async def add_single(name: str, source: str, comment: str | None = None):
    return add_layer(name, source, comment=comment)


@api.put("/single")
async def update_single(
    layer_id: str, name: str | None = None, source: str | None = None, comment: str | None = None
):
    update_layer(layer_id, name=name, source=source, comment=comment)


@api.get("/multiple")
async def read_multiple(layer_ids: Annotated[list[str] | None, Query()] = None):
    if layer_ids:
        return [asdict(x) for x in read_layers(layer_ids=layer_ids)]
    layers = read_layers()
    response = []
    for _id, layer in layers.items():
        layer.id = _id
        response.append(asdict(layer))
    return response
