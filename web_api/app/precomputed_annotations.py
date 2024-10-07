"""
This file provides web API endpoints for manipulating annotations stored in
precomputed (i.e. Neuroglancer) file format.  Design reference:
https://github.com/ZettaAI/zetta_utils/issues/797
"""

from typing import Annotated

# pylint: disable=all # type: ignore
from attrs import asdict
from fastapi import FastAPI, Query, Request
from neuroglancer.viewer_state import LineAnnotation

from zetta_utils.db_annotations import precomp_annotations
from zetta_utils.db_annotations.annotation import AnnotationDBEntry, NgAnnotation
from zetta_utils.db_annotations.precomp_annotations import (
    AnnotationLayer,
    build_annotation_layer,
)
from zetta_utils.geometry import Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex

from .utils import generic_exception_handler

api = FastAPI()


@api.exception_handler(Exception)
async def generic_handler(request: Request, exc: Exception):
    return generic_exception_handler(request, exc)


@api.get("/annotations")
async def read_in_bounds(
    path: Annotated[str, Query()],
    bbox_start: Annotated[tuple[int, int, int], Query()],
    bbox_end: Annotated[tuple[int, int, int], Query()],
    resolution: Annotated[tuple[float, float, float], Query()],
):
    """
    This endpoint retrieves all lines within the given bounds.
    """
    index = VolumetricIndex.from_coords(bbox_start, bbox_end, Vec3D(*resolution))
    layer = build_annotation_layer(path, mode="read")
    response = []
    for line in layer.read_in_bounds(index):
        annotation = AnnotationDBEntry(
            id=line.id,
            layer_group="",
            collection="",
            comment="",
            tags=[],
            ng_annotation=LineAnnotation(pointA=line.start, pointB=line.end),
        )
        response.append(annotation.to_dict())
    return response


@api.put("/annotations")
async def add_multiple(
    annotations: list[dict],
    path: Annotated[str, Query()],
    bbox_start: Annotated[tuple[int, int, int], Query()],
    bbox_end: Annotated[tuple[int, int, int], Query()],
    resolution: Annotated[tuple[float, float, float], Query()],
):
    """
    The PUT endpoint replaces all data in the given file (which may or
    may not exist yet) with the given new set of lines.
    """
    lines = []
    for entry in annotations:
        annotation = AnnotationDBEntry.from_dict(entry["id"], entry)
        line = annotation.ng_annotation
        lines.append(
            precomp_annotations.LineAnnotation(int(annotation.id), line.point_a, line.point_b)
        )
    index = VolumetricIndex.from_coords(bbox_start, bbox_end, Vec3D(*resolution))
    layer = build_annotation_layer(path, index=index, mode="replace")
    layer.write_annotations(lines)


@api.delete("/annotations")
async def delete_multiple(
    path: Annotated[str, Query()],
):
    """
    The DELETE endpoint deletes the specified precomputed annotation
    file.  Use with caution.
    """
    layer = build_annotation_layer(path, mode="write")
    layer.delete()
