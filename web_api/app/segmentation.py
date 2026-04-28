# pylint: disable=import-error
import asyncio
import gzip
import io
import json
import struct
import threading
import time
from pathlib import Path

import cutie.config as cutie_config
import numpy as np
import torch
from cutie.inference.inference_core import InferenceCore
from cutie.inference.utils.args_utils import get_dataset_cfg
from cutie.model.cutie import CUTIE
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from google.cloud import storage
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import open_dict

from .utils import generic_exception_handler

WEIGHTS_DIR = Path.home() / ".cache" / "cutie" / "weights"
GCS_WEIGHTS_PREFIX = "gs://zetta_ws/models/cutie"
WEIGHT_FILES = ["cutie-base-mega.pth", "coco_lvis_h18_itermask.pth"]

_cutie_model = None
_cutie_lock = threading.Lock()


def _ensure_weights() -> Path:
    """Download Cutie weights from GCS if not present locally."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    bucket_name = GCS_WEIGHTS_PREFIX.removeprefix("gs://").split("/", 1)[0]
    prefix = GCS_WEIGHTS_PREFIX.removeprefix(f"gs://{bucket_name}/")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for fname in WEIGHT_FILES:
        local_path = WEIGHTS_DIR / fname
        if not local_path.exists():
            blob_name = f"{prefix}/{fname}"
            print(f"Downloading gs://{bucket_name}/{blob_name}...")
            bucket.blob(blob_name).download_to_filename(str(local_path))
            print(f"  Saved to {local_path}")
    return WEIGHTS_DIR


def _load_cutie_model(device: str):
    """Load the Cutie model with weights from local cache."""
    weight_dir = _ensure_weights()

    GlobalHydra.instance().clear()

    config_dir = str(Path(cutie_config.__path__[0]))
    with initialize_config_dir(version_base="1.3.2", config_dir=config_dir):
        cfg = compose(config_name="eval_config")

    with open_dict(cfg):
        cfg["weights"] = str(weight_dir / "cutie-base-mega.pth")
        # Single-step stateless propagation: only 1 memory frame ever exists.
        # Default top_k=30 breaks on small crops where feature-map patch count < 30.
        cfg["top_k"] = 1
    get_dataset_cfg(cfg)

    model = CUTIE(cfg).to(device).eval()
    model_weights = torch.load(cfg.weights, map_location=device, weights_only=True)
    model.load_weights(model_weights)

    return model


def _get_cutie_model():
    """Lazy singleton: loads Cutie model on first call, reuses thereafter."""
    global _cutie_model  # pylint: disable=global-statement
    if _cutie_model is not None:
        return _cutie_model
    with _cutie_lock:
        if _cutie_model is not None:
            return _cutie_model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[segmentation] Loading Cutie model on {device}...")
        t0 = time.time()
        _cutie_model = _load_cutie_model(device)
        print(f"[segmentation] Cutie model loaded in {time.time() - t0:.2f}s")
        return _cutie_model


_gpu_semaphore = asyncio.Semaphore(1)

api = FastAPI()


@api.exception_handler(Exception)
async def generic_handler(request: Request, exc: Exception):
    return generic_exception_handler(request, exc)


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


async def _parse_propagate_request(request: Request):
    """Parse multipart/form-data for propagate_mask endpoint.

    Returns (current_image_np, mask_np, image_np, height, width).
    """
    form = await request.form()

    for field in ("metadata", "current-image-data", "mask-data", "image-data"):
        if field not in form:
            raise HTTPException(status_code=400, detail=f"Missing required field: '{field}'")

    metadata_str = await _read_form_field_str(form, "metadata")
    try:
        metadata = json.loads(metadata_str)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in 'metadata': {e}") from e

    missing = [k for k in ("height", "width", "mask_dtype", "image_dtype") if k not in metadata]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required metadata fields: {missing}",
        )

    height = metadata["height"]
    width = metadata["width"]

    if height <= 0 or width <= 0:
        raise HTTPException(status_code=400, detail="Dimensions must be positive")

    mask_dtype = np.dtype(metadata["mask_dtype"])
    image_dtype = np.dtype(metadata["image_dtype"])

    current_image_bytes = await _read_form_field_bytes(form, "current-image-data")
    mask_bytes = await _read_form_field_bytes(form, "mask-data")
    image_bytes = await _read_form_field_bytes(form, "image-data")

    expected_image_size = height * width * image_dtype.itemsize
    expected_mask_size = height * width * mask_dtype.itemsize

    if len(current_image_bytes) != expected_image_size:
        raise HTTPException(
            status_code=400,
            detail=f"'current-image-data' size mismatch: expected {expected_image_size} bytes "
            f"for {height}x{width} {image_dtype}, got {len(current_image_bytes)} bytes",
        )
    if len(mask_bytes) != expected_mask_size:
        raise HTTPException(
            status_code=400,
            detail=f"'mask-data' size mismatch: expected {expected_mask_size} bytes "
            f"for {height}x{width} {mask_dtype}, got {len(mask_bytes)} bytes",
        )
    if len(image_bytes) != expected_image_size:
        raise HTTPException(
            status_code=400,
            detail=f"'image-data' size mismatch: expected {expected_image_size} bytes "
            f"for {height}x{width} {image_dtype}, got {len(image_bytes)} bytes",
        )

    current_image_np = np.frombuffer(current_image_bytes, dtype=image_dtype).reshape(height, width)
    mask_np = np.frombuffer(mask_bytes, dtype=mask_dtype).reshape(height, width)
    image_np = np.frombuffer(image_bytes, dtype=image_dtype).reshape(height, width)

    return current_image_np, mask_np, image_np, height, width


def _to_frame_tensor(img_np: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert grayscale [H, W] uint8 image to 3-channel [3, H, W] float32 tensor."""
    frame = np.stack([img_np] * 3, axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(frame).to(device)


def _run_propagation(
    current_image_np: np.ndarray,
    mask_np: np.ndarray,
    image_np: np.ndarray,
    cancel_event: threading.Event,
) -> np.ndarray:
    """Run single-step Cutie mask propagation. Runs in thread pool."""
    cutie = _get_cutie_model()
    device = next(cutie.parameters()).device
    processor = InferenceCore(cutie, cfg=cutie.cfg)

    object_ids = [int(v) for v in np.unique(mask_np) if v != 0]

    t0 = time.time()
    with torch.inference_mode():
        current_tensor = _to_frame_tensor(current_image_np, device)
        mask_tensor = torch.from_numpy(mask_np.astype(np.int64)).to(device)
        processor.step(current_tensor, mask_tensor, objects=object_ids)

        if cancel_event.is_set():
            return np.zeros(mask_np.shape, dtype=np.uint8)

        next_tensor = _to_frame_tensor(image_np, device)
        output_prob = processor.step(next_tensor)
        pred = output_prob.argmax(0).cpu().numpy().astype(np.uint8)

    print(f"[segmentation] propagation inference: {time.time() - t0:.2f}s")
    return pred


def _build_binary_response(mask_np: np.ndarray, compress: bool):
    header = {"mask_shape": list(mask_np.shape)}
    header_json = json.dumps(header).encode()

    buf = io.BytesIO()
    buf.write(struct.pack("<I", len(header_json)))
    buf.write(header_json)
    buf.write(mask_np.tobytes())
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


@api.post("/propagate_mask")
async def propagate_mask(request: Request):
    """Single-step stateless mask propagation using Cutie VOS.

    Takes a current image, its segmentation mask, and the next image slice.
    Runs Cutie inference and returns the predicted mask as binary.
    """
    current_image_np, mask_np, image_np, height, width = await _parse_propagate_request(request)

    print(
        f"[segmentation] propagate_mask: {height}x{width}, "
        f"mask non-zero: {np.count_nonzero(mask_np)}"
    )

    cancel_event = threading.Event()

    async with _gpu_semaphore:
        if await request.is_disconnected():
            print("[segmentation] Client disconnected while waiting in queue")
            return Response(status_code=499)

        compute_task = asyncio.get_running_loop().run_in_executor(
            None, _run_propagation, current_image_np, mask_np, image_np, cancel_event
        )

        while not compute_task.done():
            if await request.is_disconnected():
                print("[segmentation] Client disconnected, cancelling computation")
                cancel_event.set()
                await compute_task
                return Response(status_code=499)
            await asyncio.sleep(0.5)

        pred_mask = await compute_task

    print(f"[segmentation] result: non-zero pixels: {np.count_nonzero(pred_mask)}")

    compress = "gzip" in request.headers.get("accept-encoding", "")
    return _build_binary_response(pred_mask, compress)
