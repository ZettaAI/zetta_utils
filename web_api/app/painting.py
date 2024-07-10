from typing import Annotated

import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import Response

from zetta_utils.geometry import Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex
from zetta_utils.layer.volumetric.cloudvol import build_cv_layer

api = FastAPI()


@api.get("/cutout")
async def read_cutout(
    path: Annotated[str, Query()],
    bbox_start: Annotated[tuple[int, int, int], Query()],
    bbox_end: Annotated[tuple[int, int, int], Query()],
    resolution: Annotated[tuple[float, float, float], Query()],
):
    index = VolumetricIndex.from_coords(bbox_start, bbox_end, Vec3D(*resolution))
    layer = build_cv_layer(path, readonly=True)
    return Response(content=layer[index].tobytes())


@api.post("/cutout")
async def write_cutout(
    path: Annotated[str, Query()],
    bbox_start: Annotated[tuple[int, int, int], Query()],
    bbox_end: Annotated[tuple[int, int, int], Query()],
    resolution: Annotated[tuple[float, float, float], Query()],
    data: bytes,
):
    index = VolumetricIndex.from_coords(bbox_start, bbox_end, Vec3D(*resolution))
    layer = build_cv_layer(path)
    shape = np.array(bbox_end) - np.array(bbox_start)
    layer[index] = np.frombuffer(data, dtype=layer.backend.dtype).reshape(shape)
