# pylint: disable=import-error
import base64
import traceback

import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

from zetta_utils.internal.alignment.manual_correspondence import (
    apply_correspondences_to_image,
)
from zetta_utils.log import get_logger

from .utils import generic_exception_handler

logger = get_logger("web_api.alignment")

api = FastAPI()

# Add GZip compression for large responses (base64-encoded tensors)
# minimum_size=1000 means responses under 1KB won't be compressed
api.add_middleware(GZipMiddleware, minimum_size=1000)


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
    rig: float = Field(
        1000,
        description="Rigidity penalty weight controlling smoothness"
    )
    lr: float = Field(1e-3, description="Learning rate for optimization")
    optimizer_type: str = Field(
        "adam", description="Optimizer to use (adam, lbfgs, sgd, adamw)"
    )


class ApplyCorrespondencesResponse(BaseModel):
    relaxed_field: str = Field(
        ..., description="Base64-encoded relaxed field (float32 binary)"
    )
    relaxed_field_shape: list[int] = Field(
        ..., description="Shape of relaxed_field array [C, H, W, 1]"
    )
    warped_image: str = Field(
        ..., description="Base64-encoded warped image (float32 binary)"
    )
    warped_image_shape: list[int] = Field(
        ..., description="Shape of warped_image array [C, H, W, 1]"
    )


@api.post("/apply_correspondences", response_model=ApplyCorrespondencesResponse)
async def apply_correspondences(request: ApplyCorrespondencesRequest):
    """Apply correspondences to image using relaxation and warping.

    This endpoint takes correspondence points and an image, creates a
    correspondence field, relaxes it using optimization, and applies the
    field to warp the image.

    The function automatically uses GPU (CUDA) if available, otherwise falls
    back to CPU. Deploy this API on GPU machines (T4/L4) for GPU acceleration
    or CPU machines for CPU processing.

    Returns base64-encoded float32 arrays with their shapes.
    """
    try:
        # Parse correspondences
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

        # Create tensor on CPU (internal alignment module uses CPU tensors)
        image_tensor = torch.tensor(
            request.image,
            dtype=torch.float32,
        )

        # Apply correspondences
        relaxed_field, warped_image = apply_correspondences_to_image(
            correspondences_dict=correspondences_dict,
            image=image_tensor,
            num_iter=request.num_iter,
            rig=request.rig,
            lr=request.lr,
            optimizer_type=request.optimizer_type,
        )

        # Convert to numpy arrays on CPU
        relaxed_field_np = relaxed_field.cpu().numpy().astype(np.float32)
        warped_image_np = warped_image.cpu().numpy().astype(np.float32)

        # Encode as base64
        relaxed_field_b64 = base64.b64encode(relaxed_field_np.tobytes()).decode()
        warped_image_b64 = base64.b64encode(warped_image_np.tobytes()).decode()

        return ApplyCorrespondencesResponse(
            relaxed_field=relaxed_field_b64,
            relaxed_field_shape=list(relaxed_field_np.shape),
            warped_image=warped_image_b64,
            warped_image_shape=list(warped_image_np.shape),
        )

    except Exception as e:
        logger.error("=== apply_correspondences endpoint FAILED ===")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception message: {str(e)}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        raise
