import gzip
from typing import Annotated

import einops
import numpy as np
from fastapi import FastAPI, Query, Request
from fastapi.responses import Response

from zetta_utils.geometry import Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex
from zetta_utils.layer.volumetric.cloudvol import build_cv_layer

api = FastAPI()


@app.get("/cutout")
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
    data = data.tobytes()
    compressed_data = gzip.compress(data)

    return Response(content=compressed_data)


@app.post("/cutout")
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
    # Decompress the gzipped data
    data = gzip.decompress(data)

    if not is_fortran:
        data_arr = np.frombuffer(data, dtype=layer.backend.dtype).reshape(shape)
    else:
        data_arr = np.frombuffer(data, dtype=layer.backend.dtype).reshape(shape[::-1])
        data_arr = einops.rearrange(data_arr, "Z Y X C -> C X Y Z")

    layer[index] = data_arr
