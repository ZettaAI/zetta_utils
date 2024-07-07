from typing import Annotated

import numpy as np
from fastapi import FastAPI, Query

from zetta_utils.geometry import Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex
from zetta_utils.layer.volumetric.cloudvol import build_cv_layer

api = FastAPI()


@api.get("/cutout")
async def read_cutout(
    path: Annotated[str, Query()],
    bbox_start: Annotated[tuple[int, int, int], Query()],
    bbox_end: Annotated[tuple[int, int, int], Query()],
    resolution: Annotated[Vec3D, Query()],
):
    index = VolumetricIndex.from_coords(bbox_start, bbox_end, resolution)
    layer = build_cv_layer(path, readonly=True)
    return layer[index].tobytes()


@api.post("/cutout")
async def write_cutout(
    path: Annotated[str, Query()],
    bbox_start: Annotated[tuple[int, int, int], Query()],
    bbox_end: Annotated[tuple[int, int, int], Query()],
    resolution: Annotated[Vec3D, Query()],
    data: bytes,
    dtype: str,
    shape: tuple[int, ...],
):
    index = VolumetricIndex.from_coords(bbox_start, bbox_end, resolution)
    layer = build_cv_layer(path)
    layer[index] = np.frombuffer(data, dtype=dtype).reshape(shape)
