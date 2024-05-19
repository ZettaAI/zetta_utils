"""Manage annotations in a DB Layer."""

from __future__ import annotations

from typing import Union, cast, overload

from neuroglancer.viewer_state import (
    AxisAlignedBoundingBoxAnnotation,
    EllipsoidAnnotation,
    LineAnnotation,
    PointAnnotation,
)

from zetta_utils.layer.db_layer import DBRowDataT, build_db_layer
from zetta_utils.layer.db_layer.datastore import DatastoreBackend
from zetta_utils.parsing.ngl_state import AnnotationKeys

from . import constants

NgAnnotation = Union[
    AxisAlignedBoundingBoxAnnotation, EllipsoidAnnotation, LineAnnotation, PointAnnotation
]


INDEXED_COLS = ("collection", "layer_group", "tags")
NON_INDEXED_COLS = (
    "comment",
    AnnotationKeys.TYPE.value,
    AnnotationKeys.POINT.value,
    AnnotationKeys.POINT_A.value,
    AnnotationKeys.POINT_B.value,
    AnnotationKeys.CENTER.value,
    AnnotationKeys.RADII.value,
)

DB_NAME = "annotations"
DB_BACKEND = DatastoreBackend(DB_NAME, project=constants.PROJECT, database=constants.DATABASE)
DB_BACKEND.exclude_from_indexes = NON_INDEXED_COLS
ANNOTATIONS_DB = build_db_layer(DB_BACKEND)


def read_annotation(annotation_id: str) -> DBRowDataT:
    idx = (annotation_id, INDEXED_COLS + NON_INDEXED_COLS)
    return ANNOTATIONS_DB[idx]


@overload
def read_annotations(*, annotation_ids: list[str]) -> list[DBRowDataT]:
    ...


@overload
def read_annotations(
    *,
    collection_ids: list[str] | None = None,
    layer_group_ids: list[str] | None = None,
    tags: list[str] | None = None,
) -> list[DBRowDataT]:
    ...


def read_annotations(
    *,
    annotation_ids: list[str] | None = None,
    collection_ids: list[str] | None = None,
    layer_group_ids: list[str] | None = None,
    tags: list[str] | None = None,
) -> list[DBRowDataT]:
    if annotation_ids:
        idx = (annotation_ids, INDEXED_COLS + NON_INDEXED_COLS)
        return ANNOTATIONS_DB[idx]

    _filter = {}
    if collection_ids:
        _filter["collection"] = collection_ids
    if layer_group_ids:
        _filter["layer_group"] = layer_group_ids
    if tags:
        _filter["tags"] = tags
    result = cast(DatastoreBackend, ANNOTATIONS_DB.backend).query(column_filter=_filter)
    return list(result.values())


def add_annotation(
    annotation: NgAnnotation,
    *,
    collection_id: str,
    layer_group_id: str,
    comment: str | None = None,
    tags: list[str] | None = None,
) -> str:
    row = annotation.to_json()
    row["collection"] = collection_id
    row["layer_group"] = layer_group_id
    row["comment"] = comment
    if tags:
        row["tags"] = list(set(tags))
    row_key = str(annotation.id)
    col_keys = INDEXED_COLS + NON_INDEXED_COLS
    ANNOTATIONS_DB[(row_key, col_keys)] = row
    return row_key


def add_annotations(
    annotations: list[NgAnnotation],
    *,
    collection_id: str,
    layer_group_id: str,
    comment: str | None = None,
    tags: list[str] | None = None,
) -> list[str]:
    rows = []
    row_keys = []
    for ann in annotations:
        row = ann.to_json()
        row["collection"] = collection_id
        row["layer_group"] = layer_group_id
        row["comment"] = comment
        if tags:
            row["tags"] = list(set(tags))
        rows.append(row)
        row_keys.append(str(ann.id))
    col_keys = INDEXED_COLS + NON_INDEXED_COLS
    ANNOTATIONS_DB[(row_keys, col_keys)] = rows
    return row_keys


def update_annotation(
    annotation_id: str,
    *,
    collection_id: str | None = None,
    layer_group_id: str | None = None,
    comment: str | None = None,
    tags: list[str] | None = None,
):
    col_keys = INDEXED_COLS + NON_INDEXED_COLS
    row: DBRowDataT = {}
    if collection_id:
        row["collection"] = collection_id
    if layer_group_id:
        row["layer_group"] = layer_group_id
    if comment:
        row["comment"] = comment
    if tags:
        row["tags"] = list(set(tags))
    ANNOTATIONS_DB[(annotation_id, col_keys)] = row


def update_annotations(
    annotation_ids: list[str],
    *,
    collection_id: str | None = None,
    layer_group_id: str | None = None,
    comment: str | None = None,
    tags: list[str] | None = None,
):
    col_keys = INDEXED_COLS + NON_INDEXED_COLS
    rows = []
    for _ in range(len(annotation_ids)):
        row: DBRowDataT = {}
        if collection_id:
            row["collection"] = collection_id
        if layer_group_id:
            row["layer_group"] = layer_group_id
        if comment:
            row["comment"] = comment
        if tags:
            row["tags"] = list(set(tags))
        rows.append(row)
    ANNOTATIONS_DB[(annotation_ids, col_keys)] = rows


def delete_annotation(annotation_id: str):
    raise NotImplementedError()


def delete_annotations(annotation_ids: list[str]):
    raise NotImplementedError()


def parse_ng_annotations(annotations_raw: list[dict]) -> list[NgAnnotation]:
    annotations = []
    for ann in annotations_raw:
        _type = ann.pop("type")
        if _type == "line":
            annotations.append(LineAnnotation(**ann))
        elif _type == "ellipsoid":
            annotations.append(EllipsoidAnnotation(**ann))
        elif _type == "point":
            annotations.append(PointAnnotation(**ann))
        else:
            annotations.append(AxisAlignedBoundingBoxAnnotation(**ann))
    return annotations
