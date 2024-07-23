from typing import Annotated

import einops
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
    permute_pattern: Annotated[str | None, Query()] = None,
):
    index = VolumetricIndex.from_coords(bbox_start, bbox_end, Vec3D(*resolution))
    layer = build_cv_layer(path, readonly=True)
    data = np.ascontiguousarray(layer[index])
    if permute_pattern:
        data = einops.rearrange(data, permute_pattern)
    return Response(content=data.tobytes())


@api.post("/cutout")
async def write_cutout(
    data: bytes,
    path: Annotated[str, Query()],
    bbox_start: Annotated[tuple[int, int, int], Query()],
    bbox_end: Annotated[tuple[int, int, int], Query()],
    resolution: Annotated[tuple[float, float, float], Query()],
    permute_pattern: Annotated[str | None, Query()] = None,
):
    index = VolumetricIndex.from_coords(bbox_start, bbox_end, Vec3D(*resolution))
    cv_kwargs = {"non_aligned_writes": True}
    layer = build_cv_layer(path, cv_kwargs=cv_kwargs)
    shape = np.array(bbox_end) - np.array(bbox_start)
    data_arr = np.frombuffer(data, dtype=layer.backend.dtype).reshape(shape)
    if permute_pattern:
        data_arr = einops.rearrange(data_arr, permute_pattern)
    layer[index] = data_arr
