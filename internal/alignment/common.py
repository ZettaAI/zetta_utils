from __future__ import annotations

from typing import Any

import torch
from typeguard import typechecked


@typechecked
def tiled_inference(
    model,
    data_in: torch.Tensor,
    tile_size: int,
    tile_pad_in: int,
    ds_factor: int = 1,
    output_dtype: Any = torch.float32,
    output_channels: int = 1,
    device: str | None = None,
    enable_torch_autocast: bool = True,
) -> torch.Tensor:

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    result = torch.zeros(
        data_in.shape[0],
        output_channels,
        data_in.shape[-2] // ds_factor,
        data_in.shape[-1] // ds_factor,
        dtype=output_dtype,
        layout=data_in.layout,
        device=data_in.device,
    )
    tile_pad_out = tile_pad_in // ds_factor

    for x in range(tile_pad_in, data_in.shape[-2] - tile_pad_in, tile_size):
        x_start = x - tile_pad_in
        x_end = x + tile_size + tile_pad_in
        for y in range(
            tile_pad_in,
            data_in.shape[-1] - tile_pad_in,
            tile_size,
        ):
            y_start = y - tile_pad_in
            y_end = y + tile_size + tile_pad_in
            tile = data_in[:, :, x_start:x_end, y_start:y_end]
            if (tile != 0).sum() > 0.0:
                with torch.autocast(device_type=device, enabled=enable_torch_autocast):
                    tile_result = model(tile)
                if tile_pad_out > 0:
                    tile_result = tile_result[
                        :,
                        :,
                        tile_pad_out:-tile_pad_out,
                        tile_pad_out:-tile_pad_out,
                    ]

                result[
                    :,
                    :,
                    x // ds_factor : x // ds_factor + tile_result.shape[-2],
                    y // ds_factor : y // ds_factor + tile_result.shape[-1],
                ] = tile_result

    return result
