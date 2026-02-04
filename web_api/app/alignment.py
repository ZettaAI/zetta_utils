# pylint: disable=import-error
import base64
import gzip
import io
import json
import struct

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
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
    rig: float = Field(
        1000,
        description="Rigidity penalty weight controlling smoothness"
    )
    lr: float = Field(1e-3, description="Learning rate for optimization")
    optimizer_type: str = Field(
        "adam", description="Optimizer to use (adam, lbfgs, sgd, adamw)"
    )
    tissue_src_mask: list[list[list[list[float]]]] | None = Field(
        None,
        description="Binary tissue mask (1=tissue, 0=non-tissue), shape (1, H, W, 1). "
        "Zeros break rigidity constraints between regions.",
    )
    tissue_tgt_mask: list[list[list[list[float]]]] | None = Field(
        None,
        description="Binary tissue mask (1=tissue, 0=non-tissue), shape (1, H, W, 1). "
        "Zeros break rigidity constraints between regions.",
    )
    defect_src_mask: list[list[list[list[float]]]] | None = Field(
        None,
        description="Binary defect mask (1=defect, 0=non-defect), shape (1, H, W, 1). "
        "Inverted tissue mask. Cannot be used together with tissue masks.",
    )
    defect_tgt_mask: list[list[list[list[float]]]] | None = Field(
        None,
        description="Binary defect mask (1=defect, 0=non-defect), shape (1, H, W, 1). "
        "Inverted tissue mask. Cannot be used together with tissue masks.",
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


def _decompress_if_gzipped(data: bytes) -> bytes:
    if data[:2] == b"\x1f\x8b":
        return gzip.decompress(data)
    return data


def _parse_json_request(body: dict, device: torch.device):
    req = ApplyCorrespondencesRequest(**body)

    correspondences_dict = {
        "lines": [
            {"start": line.start, "end": line.end, "id": line.id}
            for line in req.correspondences_dict.lines
        ]
    }

    image_tensor = torch.tensor(req.image, dtype=torch.float32, device=device)

    has_tissue = req.tissue_src_mask is not None or req.tissue_tgt_mask is not None
    has_defect = req.defect_src_mask is not None or req.defect_tgt_mask is not None
    if has_tissue and has_defect:
        raise ValueError("Cannot use both tissue and defect masks simultaneously")

    src_mask_tensor = None
    tgt_mask_tensor = None

    if req.tissue_src_mask is not None:
        src_mask_tensor = torch.tensor(req.tissue_src_mask, dtype=torch.float32, device=device)
    elif req.defect_src_mask is not None:
        defect_mask = torch.tensor(req.defect_src_mask, dtype=torch.float32, device=device)
        src_mask_tensor = 1.0 - defect_mask

    if req.tissue_tgt_mask is not None:
        tgt_mask_tensor = torch.tensor(req.tissue_tgt_mask, dtype=torch.float32, device=device)
    elif req.defect_tgt_mask is not None:
        defect_mask = torch.tensor(req.defect_tgt_mask, dtype=torch.float32, device=device)
        tgt_mask_tensor = 1.0 - defect_mask

    return correspondences_dict, image_tensor, src_mask_tensor, tgt_mask_tensor, {
        "num_iter": req.num_iter,
        "rig": req.rig,
        "lr": req.lr,
        "optimizer_type": req.optimizer_type,
    }


async def _read_form_field_bytes(form, field_name: str) -> bytes:
    value = form[field_name]
    if hasattr(value, "read"):
        return await value.read()
    if isinstance(value, str):
        raise HTTPException(
            status_code=400,
            detail=f"'{field_name}' must be sent as a file upload, not a plain text form field",
        )
    if isinstance(value, bytes):
        return value
    raise HTTPException(
        status_code=400,
        detail=f"'{field_name}' must be sent as a file upload, not a plain form value",
    )


async def _read_form_field_str(form, field_name: str) -> str:
    value = form[field_name]
    if hasattr(value, "read"):
        return (await value.read()).decode()
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _read_tensor_from_bytes(
    data: bytes, shape: list[int], field_name: str
) -> np.ndarray:
    data = _decompress_if_gzipped(data)
    expected_size = int(np.prod(shape)) * 4
    if len(data) != expected_size:
        raise HTTPException(
            status_code=400,
            detail=f"'{field_name}' size mismatch: expected {expected_size} bytes "
            f"for shape {shape}, got {len(data)} bytes",
        )
    return np.frombuffer(data, dtype=np.float32).reshape(shape)


async def _parse_multipart_request(request: Request, device: torch.device):
    form = await request.form()

    if "metadata" not in form:
        raise HTTPException(status_code=400, detail="Missing required field: 'metadata'")
    if "image-data" not in form:
        raise HTTPException(status_code=400, detail="Missing required field: 'image-data'")

    metadata_str = await _read_form_field_str(form, "metadata")
    try:
        metadata = json.loads(metadata_str)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in 'metadata': {e}") from e

    missing = [k for k in ("correspondences", "image_shape") if k not in metadata]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required metadata fields: {missing}",
        )

    correspondences_dict = metadata["correspondences"]

    image_bytes = await _read_form_field_bytes(form, "image-data")
    image_np = _read_tensor_from_bytes(image_bytes, metadata["image_shape"], "image-data")
    image_tensor = torch.tensor(image_np, dtype=torch.float32, device=device)

    src_mask_tensor = None
    tgt_mask_tensor = None
    mask_type = metadata.get("mask_type")

    if "src-mask-data" in form:
        if "src_mask_shape" not in metadata:
            raise HTTPException(
                status_code=400,
                detail="'src_mask_shape' required in metadata when 'src-mask-data' is provided",
            )
        src_bytes = await _read_form_field_bytes(form, "src-mask-data")
        src_np = _read_tensor_from_bytes(src_bytes, metadata["src_mask_shape"], "src-mask-data")
        src_mask_tensor = torch.tensor(src_np, dtype=torch.float32, device=device)
        if mask_type == "defect":
            src_mask_tensor = 1.0 - src_mask_tensor

    if "tgt-mask-data" in form:
        if "tgt_mask_shape" not in metadata:
            raise HTTPException(
                status_code=400,
                detail="'tgt_mask_shape' required in metadata when 'tgt-mask-data' is provided",
            )
        tgt_bytes = await _read_form_field_bytes(form, "tgt-mask-data")
        tgt_np = _read_tensor_from_bytes(tgt_bytes, metadata["tgt_mask_shape"], "tgt-mask-data")
        tgt_mask_tensor = torch.tensor(tgt_np, dtype=torch.float32, device=device)
        if mask_type == "defect":
            tgt_mask_tensor = 1.0 - tgt_mask_tensor

    return correspondences_dict, image_tensor, src_mask_tensor, tgt_mask_tensor, {
        "num_iter": metadata.get("num_iter", 200),
        "rig": metadata.get("rig", 1000),
        "lr": metadata.get("lr", 1e-3),
        "optimizer_type": metadata.get("optimizer_type", "adam"),
    }


def _build_json_response(relaxed_field_np: np.ndarray, warped_image_np: np.ndarray):
    relaxed_field_b64 = base64.b64encode(relaxed_field_np.tobytes()).decode()
    warped_image_b64 = base64.b64encode(warped_image_np.tobytes()).decode()
    return ApplyCorrespondencesResponse(
        relaxed_field=relaxed_field_b64,
        relaxed_field_shape=list(relaxed_field_np.shape),
        warped_image=warped_image_b64,
        warped_image_shape=list(warped_image_np.shape),
    )


def _build_binary_response(
    relaxed_field_np: np.ndarray, warped_image_np: np.ndarray, compress: bool
):
    header_json = json.dumps({
        "relaxed_field_shape": list(relaxed_field_np.shape),
        "warped_image_shape": list(warped_image_np.shape),
    }).encode()

    buf = io.BytesIO()
    buf.write(struct.pack("<I", len(header_json)))
    buf.write(header_json)
    buf.write(relaxed_field_np.tobytes())
    buf.write(warped_image_np.tobytes())
    payload = buf.getvalue()

    if compress:

        def chunked_compress(data: bytes):
            with io.BytesIO() as buffer:
                with gzip.GzipFile(fileobj=buffer, mode="wb") as gzip_file:
                    gzip_file.write(data)
                buffer.seek(0)
                while chunk := buffer.read(64 * 1024):
                    yield chunk

        return StreamingResponse(
            chunked_compress(payload),
            media_type="application/gzip",
        )

    return Response(content=payload, media_type="application/octet-stream")


def _wants_binary_response(request: Request) -> bool:
    accept = request.headers.get("accept", "")
    fmt = request.headers.get("x-alignment-response-format", "")
    return "application/octet-stream" in accept or fmt == "binary-v1"


@api.post("/apply_correspondences")
async def apply_correspondences(request: Request):
    """Apply correspondences to image using relaxation and warping.

    This endpoint takes correspondence points and an image, creates a
    correspondence field, relaxes it using optimization, and applies the
    field to warp the image.

    The function automatically uses GPU (CUDA) if available, otherwise falls
    back to CPU. Deploy this API on GPU machines (T4/L4) for GPU acceleration
    or CPU machines for CPU processing.

    Supports two request formats:
    - application/json: JSON body with base64-encoded arrays (backward compatible)
    - multipart/form-data: Binary tensor data with JSON metadata (efficient)

    Returns base64-encoded JSON or raw binary depending on Accept header.
    """
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"CUDA is available. Using GPU: {gpu_name}")
    else:
        print("CUDA is not available. Using CPU")

    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type:
        correspondences_dict, image_tensor, src_mask_tensor, tgt_mask_tensor, params = (
            await _parse_multipart_request(request, device)
        )
    else:
        body = await request.json()
        correspondences_dict, image_tensor, src_mask_tensor, tgt_mask_tensor, params = (
            _parse_json_request(body, device)
        )

    relaxed_field, warped_image = apply_correspondences_to_image(
        correspondences_dict=correspondences_dict,
        image=image_tensor,
        src_mask=src_mask_tensor,
        tgt_mask=tgt_mask_tensor,
        **params,
    )

    relaxed_field_np = relaxed_field.cpu().numpy().astype(np.float32)
    warped_image_np = warped_image.cpu().numpy().astype(np.float32)

    if _wants_binary_response(request):
        compress = "gzip" in request.headers.get("accept-encoding", "")
        return _build_binary_response(relaxed_field_np, warped_image_np, compress)

    return _build_json_response(relaxed_field_np, warped_image_np)
