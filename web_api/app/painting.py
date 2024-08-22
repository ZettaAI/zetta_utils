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
from zetta_utils.log import get_logger

logger = get_logger("zetta_ai")
api = FastAPI()


@api.get("/cutout")
async def read_cutout(
    path: Annotated[str, Query()],
    bbox_start: Annotated[tuple[int, int, int], Query()],
    bbox_end: Annotated[tuple[int, int, int], Query()],
    resolution: Annotated[tuple[float, float, float], Query()],
    is_fortran: Annotated[bool, Query()] = True,
):
    index = VolumetricIndex.from_coords(bbox_start, bbox_end, Vec3D(*resolution))
    logger.info(f"Index: {index}")
    layer = build_cv_layer(path, readonly=True)

    data = np.ascontiguousarray(layer[index])
    logger.info(f"Data shape: {data.shape}")
    if is_fortran:
        data = einops.rearrange(data, "C X Y Z -> Z Y X C")
        logger.info(f"Data shape after rearrange: {data.shape}")
    data_bytes = data.tobytes()
    logger.info(f"Num bytes: {len(data_bytes)}")
    compressed_data = gzip.compress(data_bytes)
    logger.info(f"Num compressed bytes: {len(data_bytes)}")

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
    logger.info(f"Index: {index}")
    cv_kwargs = {"non_aligned_writes": True}
    layer = build_cv_layer(path, cv_kwargs=cv_kwargs)
    logger.info(f"Non aligned writes enabled: {layer.backend.enforce_chunk_aligned_writes}")
    shape = [layer.backend.num_channels, *(np.array(bbox_end) - np.array(bbox_start))]

    logger.info(f"Expected shape: {shape}")
    data = await request.body()
    logger.info(f"Compressed data len: {len(data)}")
    # Decompress the gzipped data
    data = gzip.decompress(data)
    logger.info(f"Decompressed data len: {len(data)}")

    if not is_fortran:
        data_arr = np.frombuffer(data, dtype=layer.backend.dtype).reshape(shape)
    else:
        data_arr = np.frombuffer(data, dtype=layer.backend.dtype).reshape(shape[::-1])
        data_arr = einops.rearrange(data_arr, "Z Y X C -> C X Y Z")

    logger.info(f"Final data shape: {data_arr.shape}")
    layer[index] = data_arr
