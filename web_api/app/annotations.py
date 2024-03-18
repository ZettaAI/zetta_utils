from typing import Annotated

from fastapi import FastAPI, Query

from zetta_utils.db_annotations.annotation import (
    add_annotation,
    add_annotations,
    parse_ng_annotations,
    read_annotation,
    read_annotations,
    update_annotation,
    update_annotations,
)

api = FastAPI()


@api.get("/single/{annotation_id}")
async def read_single(annotation_id: str):
    return read_annotation(annotation_id)


@api.post("/single")
async def add_single(
    annotation: dict,
    collection_id: str,
    layer_group_id: str,
    comment: str | None = None,
    tags: list[str] | None = None,
):
    return add_annotation(
        parse_ng_annotations([annotation])[0],
        collection_id=collection_id,
        layer_group_id=layer_group_id,
        comment=comment,
        tags=tags,
    )


@api.put("/single")
async def update_single(
    annotation_id: str,
    collection_id: str | None = None,
    layer_group_id: str | None = None,
    comment: str | None = None,
    tags: list[str] | None = None,
):
    update_annotation(
        annotation_id,
        collection_id=collection_id,
        layer_group_id=layer_group_id,
        comment=comment,
        tags=tags,
    )


@api.get("/multiple")
async def read_multiple(
    annotation_ids: Annotated[list[str] | None, Query()] = None,
    collection_ids: Annotated[list[str] | None, Query()] = None,
    layer_group_ids: Annotated[list[str] | None, Query()] = None,
    tags: Annotated[list[str] | None, Query()] = None,
):
    if annotation_ids:
        return read_annotations(annotation_ids=annotation_ids)
    return read_annotations(
        collection_ids=collection_ids, layer_group_ids=layer_group_ids, tags=tags
    )


@api.post("/multiple")
async def add_multiple(
    annotations: list[dict],
    collection_id: str,
    layer_group_id: str,
    comment: str | None = None,
    tags: list[str] | None = None,
):
    return add_annotations(
        parse_ng_annotations(annotations),
        collection_id=collection_id,
        layer_group_id=layer_group_id,
        comment=comment,
        tags=tags,
    )


@api.put("/multiple")
async def update_multiple(
    annotation_ids: list[str],
    collection_id: str | None = None,
    layer_group_id: str | None = None,
    comment: str | None = None,
    tags: list[str] | None = None,
):
    update_annotations(
        annotation_ids,
        collection_id=collection_id,
        layer_group_id=layer_group_id,
        comment=comment,
        tags=tags,
    )
