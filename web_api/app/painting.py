# pylint: disable=all # type: ignore
import gzip
from typing import Annotated

import einops
import numpy as np
from fastapi import FastAPI, Query, Request
from fastapi.responses import Response

from zetta_utils.geometry import Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex
from zetta_utils.layer.volumetric.cloudvol import build_cv_layer

from .utils import generic_exception_handler

api = FastAPI()


@api.exception_handler(Exception)
async def generic_handler(request: Request, exc: Exception):
    return generic_exception_handler(request, exc)


@api.get("/cutout")
async def read_cutout(
    path: Annotated[str, Query()],
    bbox_start: Annotated[tuple[int, int, int], Query()],
    bbox_end: Annotated[tuple[int, int, int], Query()],
    resolution: Annotated[tuple[float, float, float], Query()],
    is_fortran: Annotated[bool, Query()] = True,
):
    index = VolumetricIndex.from_coords(bbox_start, bbox_end, Vec3D(*resolution))
    layer = build_cv_layer(path, readonly=True)

    data = np.ascontiguousarray(layer[index])
    if is_fortran:
        data = einops.rearrange(data, "C X Y Z -> Z Y X C")
    data_bytes = data.tobytes()
    compressed_data = gzip.compress(data_bytes)
    return Response(content=compressed_data)


@api.post("/cutout")
async def write_cutout(
    request: Request,
    path: Annotated[str, Query()],
    bbox_start: Annotated[tuple[int, int, int], Query()],
    bbox_end: Annotated[tuple[int, int, int], Query()],
    resolution: Annotated[tuple[float, float, float], Query()],
    is_fortran: Annotated[bool, Query()] = True,
):
    index = VolumetricIndex.from_coords(bbox_start, bbox_end, Vec3D(*resolution))
    cv_kwargs = {"non_aligned_writes": True}
    layer = build_cv_layer(path, cv_kwargs=cv_kwargs)
    shape = [layer.backend.num_channels, *(np.array(bbox_end) - np.array(bbox_start))]

    data = await request.body()
    data = gzip.decompress(data)

    if not is_fortran:
        data_arr = np.frombuffer(data, dtype=layer.backend.dtype).reshape(shape)
    else:
        data_arr = np.frombuffer(data, dtype=layer.backend.dtype).reshape(shape[::-1])
        data_arr = einops.rearrange(data_arr, "Z Y X C -> C X Y Z")
    layer[index] = data_arr
