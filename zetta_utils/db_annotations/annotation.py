"""Manage annotations in a DB Layer."""

from __future__ import annotations

import copy
import json
import time
import uuid
from typing import Any, Mapping, Sequence, Union, cast, overload

import attrs
from neuroglancer.viewer_state import (
    AxisAlignedBoundingBoxAnnotation,
    EllipsoidAnnotation,
    LineAnnotation,
    PointAnnotation,
)
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.geometry.bbox import BBox3D
from zetta_utils.geometry.vec import Vec3D
from zetta_utils.layer.db_layer import DBRowDataT
from zetta_utils.layer.db_layer.firestore import build_firestore_layer
from zetta_utils.parsing.ngl_state import AnnotationKeys

from . import constants

NgAnnotation = Union[
    AxisAlignedBoundingBoxAnnotation, EllipsoidAnnotation, LineAnnotation, PointAnnotation
]


INDEXED_COLS = ("collection", "layer_group", "tags", "created_at", "modified_at")
NON_INDEXED_COLS = (
    "comment",
    AnnotationKeys.TYPE.value,
    AnnotationKeys.POINT.value,
    AnnotationKeys.POINT_A.value,
    AnnotationKeys.POINT_B.value,
    AnnotationKeys.CENTER.value,
    AnnotationKeys.RADII.value,
    "selected_segments",
)

DB_NAME = "annotations"
ANNOTATIONS_DB = build_firestore_layer(
    DB_NAME,
    project=constants.PROJECT,
    database=constants.DATABASE,
)


@attrs.mutable
class AnnotationDBEntry:
    id: str
    layer_group: str
    collection: str
    ng_annotation: NgAnnotation
    comment: str
    tags: list[str]
    selected_segments: Mapping[str, Sequence[int]]
    created_at: float | None = None
    modified_at: float | None = None

    @staticmethod
    def from_dict(annotation_id: str, raw_dict: dict[str, Any]) -> AnnotationDBEntry:
        raw_with_defaults: dict[str, Any] = {"tags": [], "selected_segments": [], **raw_dict}
        shape_dict = copy.deepcopy(raw_with_defaults)
        shape_dict.pop("layer_group", None)
        shape_dict.pop("collection", None)
        shape_dict.pop("comment", None)
        shape_dict.pop("tags", None)
        shape_dict.pop("selected_segments", None)
        ng_annotation = parse_ng_annotations([shape_dict])[0]

        result = AnnotationDBEntry(
            id=annotation_id,
            layer_group=raw_with_defaults.get("layer_group", ""),
            collection=raw_with_defaults.get("collection", ""),
            comment=raw_with_defaults.get("comment", ""),
            tags=raw_with_defaults["tags"],
            selected_segments=raw_with_defaults["selected_segments"],
            ng_annotation=ng_annotation,
            created_at=raw_with_defaults.get("created_at"),
            modified_at=raw_with_defaults.get("modified_at"),
        )
        return result

    def to_dict(self):
        result = dict(self.ng_annotation.to_json())
        result.update(attrs.asdict(self))
        del result["ng_annotation"]
        return result


def read_annotation(annotation_id: str) -> AnnotationDBEntry:
    idx = (annotation_id, INDEXED_COLS + NON_INDEXED_COLS)
    raw_dict = ANNOTATIONS_DB[idx]

    return AnnotationDBEntry.from_dict(annotation_id=annotation_id, raw_dict=raw_dict)


@overload
def read_annotations(*, annotation_ids: list[str]) -> list[AnnotationDBEntry]:
    ...


@overload
def read_annotations(
    *,
    collection_ids: list[str] | None = None,
    layer_group_ids: list[str] | None = None,
    tags: list[str] | None = None,
    union: bool = True,
) -> dict[str, AnnotationDBEntry]:
    ...


def read_annotations(
    *,
    annotation_ids=None,
    collection_ids=None,
    layer_group_ids=None,
    tags=None,
    union: bool = True,
):
    if annotation_ids:
        idx = (annotation_ids, INDEXED_COLS + NON_INDEXED_COLS)
        result_raw = ANNOTATIONS_DB[idx]
        return [
            AnnotationDBEntry.from_dict(annotation_ids[i], result_raw[i])
            for i in range(len(annotation_ids))
        ]
    else:
        _filter = {}
        if collection_ids:
            _filter["collection"] = collection_ids
        if layer_group_ids:
            _filter["layer_group"] = layer_group_ids
        if tags:
            _filter["-tags"] = tags
        result_raw = ANNOTATIONS_DB.backend.query(column_filter=_filter, union=union)
        return {k: AnnotationDBEntry.from_dict(k, cast(dict, v)) for k, v in result_raw.items()}


def _make_annotation_row(
    annotation: NgAnnotation,
    collection_id: str,
    layer_group_id: str,
    comment: str | None,
    selected_segments: Mapping[str, Sequence[str]],
    tags: list[str] | None,
) -> tuple[str, dict]:
    row = annotation.to_json()
    row["collection"] = collection_id
    row["layer_group"] = layer_group_id
    row["comment"] = comment

    row_str = json.dumps(row, sort_keys=True)
    annotation_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, row_str))
    row["created_at"] = time.time()
    if tags:
        row["tags"] = list(set(tags))
    row["selected_segments"] = selected_segments
    return annotation_id, row


def add_annotation(
    annotation: NgAnnotation,
    *,
    collection_id: str,
    layer_group_id: str,
    comment: str | None = None,
    selected_segments: Mapping[str, Sequence[str]] | None = None,
    tags: list[str] | None = None,
) -> str:
    annotation_id, row = _make_annotation_row(
        annotation,
        collection_id=collection_id,
        layer_group_id=layer_group_id,
        comment=comment,
        selected_segments=selected_segments if selected_segments else {},
        tags=tags,
    )
    col_keys = INDEXED_COLS + NON_INDEXED_COLS
    ANNOTATIONS_DB[(annotation_id, col_keys)] = row
    return annotation_id


def add_annotations(
    annotations: list[NgAnnotation],
    *,
    collection_id: str,
    layer_group_id: str,
    comment: str | None = None,
    selected_segments: Sequence[Mapping[str, Sequence[str]]] | None = None,
    tags: list[str] | None = None,
) -> list[str]:
    rows = []
    annotation_ids = []
    for i, annotation in enumerate(annotations):
        segments: Mapping[str, Sequence[str]] = {}
        if selected_segments is not None:
            segments = selected_segments[i]
        annotation_id, row = _make_annotation_row(
            annotation,
            collection_id=collection_id,
            layer_group_id=layer_group_id,
            comment=comment,
            selected_segments=segments,
            tags=tags,
        )
        rows.append(row)
        annotation_ids.append(annotation_id)
    col_keys = INDEXED_COLS + NON_INDEXED_COLS
    ANNOTATIONS_DB[(annotation_ids, col_keys)] = rows
    return annotation_ids


def update_annotation(
    annotation_id: str,
    *,
    collection_id: str | None = None,
    layer_group_id: str | None = None,
    selected_segments: Mapping[str, Sequence[int]] | None = None,
    comment: str | None = None,
    tags: list[str] | None = None,
):
    col_keys = INDEXED_COLS + NON_INDEXED_COLS
    row: DBRowDataT = {"modified_at": time.time()}
    if collection_id is not None:
        row["collection"] = collection_id
    if layer_group_id is not None:
        row["layer_group"] = layer_group_id
    if comment is not None:
        row["comment"] = comment
    if tags is not None:
        row["tags"] = list(set(tags))

    if selected_segments is not None:
        row["selected_segments"] = {k: list(v) for k, v in selected_segments.items()}
    ANNOTATIONS_DB[(annotation_id, col_keys)] = row


def update_annotations(
    annotation_ids: list[str],
    *,
    collection_id: str | None = None,
    layer_group_id: str | None = None,
    comment: str | None = None,
    selected_segments: Sequence[Mapping[str, Sequence[int]]] | None = None,
    tags: list[str] | None = None,
):
    col_keys = INDEXED_COLS + NON_INDEXED_COLS
    rows = []
    for i in range(len(annotation_ids)):
        row: DBRowDataT = {"modified_at": time.time()}
        if collection_id is not None:
            row["collection"] = collection_id
        if layer_group_id is not None:
            row["layer_group"] = layer_group_id
        if comment is not None:
            row["comment"] = comment
        if tags is not None:
            row["tags"] = list(set(tags))
        if selected_segments is not None:
            row["selected_segments"] = {k: list(v) for k, v in selected_segments[i].items()}
        rows.append(row)
    ANNOTATIONS_DB[(annotation_ids, col_keys)] = rows


def delete_annotation(annotation_id: str):
    del ANNOTATIONS_DB[annotation_id]


def delete_annotations(annotation_ids: list[str]):
    del ANNOTATIONS_DB[annotation_ids]


def parse_ng_annotations(annotations_raw: list[dict]) -> list[NgAnnotation]:
    annotations = []
    for ann in annotations_raw:
        _type = ann.pop("type")
        ann.pop("created_at", None)
        ann.pop("modified_at", None)
        if _type == "line":
            annotations.append(LineAnnotation(**ann))
        elif _type == "ellipsoid":
            annotations.append(EllipsoidAnnotation(**ann))
        elif _type == "point":
            annotations.append(PointAnnotation(**ann))
        else:
            annotations.append(AxisAlignedBoundingBoxAnnotation(**ann))
    return annotations


@typechecked
@builder.register("add_bbox_annotation")
def add_bbox_annotation(
    bbox: BBox3D,
    *,
    collection_id: str,
    layer_group_id: str,
    comment: str | None = None,
    selected_segments: Mapping[str, Sequence[str]] | None = None,
    tags: list[str] | None = None,
) -> str:  # pragma: no cover # delegation
    return add_annotation(
        AxisAlignedBoundingBoxAnnotation(
            point_a=bbox.start,
            point_b=bbox.end,
        ),
        collection_id=collection_id,
        layer_group_id=layer_group_id,
        selected_segments=selected_segments,
        comment=comment,
        tags=tags,
    )


@typechecked
@builder.register("add_point_annotation")
def add_point_annotation(
    coord: Vec3D,
    *,
    collection_id: str,
    layer_group_id: str,
    comment: str | None = None,
    selected_segments: Mapping[str, Sequence[str]] | None = None,
    tags: list[str] | None = None,
) -> str:  # pragma: no cover # delegation
    return add_annotation(
        PointAnnotation(
            point=list(coord),
        ),
        collection_id=collection_id,
        layer_group_id=layer_group_id,
        selected_segments=selected_segments,
        comment=comment,
        tags=tags,
    )
