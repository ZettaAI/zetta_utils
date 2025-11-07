# pylint: disable=all # type: ignore
from typing import Annotated

import torch
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from zetta_utils.internal.alignment.manual_correspondence import (
    apply_correspondences_to_image,
)

from .utils import generic_exception_handler

api = FastAPI()


@api.exception_handler(Exception)
async def generic_handler(request: Request, exc: Exception):
    return generic_exception_handler(request, exc)


class CorrespondenceLine(BaseModel):
    start: list[float] = Field(..., description="Start point [y, x]")
    end: list[float] = Field(..., description="End point [y, x]")
    id: str = Field(..., description="Line ID")


class CorrespondencesDict(BaseModel):
    lines: list[CorrespondenceLine] = Field(
        ..., description="List of correspondence lines"
    )


class ApplyCorrespondencesRequest(BaseModel):
    correspondences_dict: CorrespondencesDict = Field(
        ..., description="Dictionary with correspondence lines"
    )
    image: list[list[list[list[float]]]] = Field(
        ..., description="Image tensor as nested list (C, H, W, 1)"
    )
    num_iter: int = Field(200, description="Number of optimization iterations")
    rig: float = Field(1000, description="Rigidity penalty weight controlling smoothness")
    lr: float = Field(1e-3, description="Learning rate for optimization")
    optimizer_type: str = Field("adam", description="Optimizer to use (adam, lbfgs, sgd, adamw)")


class ApplyCorrespondencesResponse(BaseModel):
    relaxed_field: list[list[list[list[float]]]] = Field(
        ..., description="Relaxed correspondence field (2, H, W, 1)"
    )
    warped_image: list[list[list[list[float]]]] = Field(
        ..., description="Warped image tensor (C, H, W, 1)"
    )


@api.post("/apply_correspondences", response_model=ApplyCorrespondencesResponse)
async def apply_correspondences(request: ApplyCorrespondencesRequest):
    """Apply correspondences to image using relaxation and warping.

    This endpoint takes correspondence points and an image, creates a correspondence field,
    relaxes it using optimization, and applies the field to warp the image.
    """
    correspondences_dict = {
        "lines": [
            {
                "start": line.start,
                "end": line.end,
                "id": line.id,
            }
            for line in request.correspondences_dict.lines
        ]
    }

    image_tensor = torch.tensor(request.image, dtype=torch.float32)

    relaxed_field, warped_image = apply_correspondences_to_image(
        correspondences_dict=correspondences_dict,
        image=image_tensor,
        num_iter=request.num_iter,
        rig=request.rig,
        lr=request.lr,
        optimizer_type=request.optimizer_type,
    )

    return ApplyCorrespondencesResponse(
        relaxed_field=relaxed_field.tolist(),
        warped_image=warped_image.tolist(),
    )
