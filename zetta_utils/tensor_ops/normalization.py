import cv2
import torch

from zetta_utils import builder
from zetta_utils.tensor_ops import convert
from zetta_utils.tensor_typing import Tensor


@builder.register("apply_clahe")
def apply_clahe(data: Tensor, clip_limit: float = 80.0, tile_grid_size: int = 64) -> torch.Tensor:
    data_torch = convert.to_torch(data)
    if not data_torch.dtype in (torch.int8, torch.uint8):
        raise NotImplementedError("CLAHE is only supported for Int8 / UInt8 tensors / arrays.")
    device = data_torch.device
    shape = data_torch.shape
    data_squeezed = data_torch.squeeze()
    if len(data_squeezed.shape) != 2:
        raise NotImplementedError(
            "CLAHE is only supported for two-dimensional (excluding"
            f" singletons) tensors / arrays: received data_squeezed of shape {shape}."
        )
    zero_mask = data_squeezed == 0
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    if data_squeezed.dtype == torch.int8:
        clahed_data_squeezed = (
            torch.tensor(
                clahe.apply((data_squeezed + 128).to("cpu", dtype=torch.uint8).numpy()),
                dtype=torch.int8,
                device=device,
            )
            - 128
        )
    else:
        assert data_squeezed.dtype == torch.uint8
        clahed_data_squeezed = torch.tensor(
            clahe.apply(data_squeezed.cpu().numpy()), dtype=torch.uint8, device=device
        )
    return torch.where(
        zero_mask, torch.zeros_like(clahed_data_squeezed), clahed_data_squeezed
    ).reshape(shape)
