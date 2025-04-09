from __future__ import annotations

import attrs
import einops
import torch
from typeguard import typechecked

from zetta_utils import builder, convnet
from zetta_utils.layer import volumetric
from zetta_utils.tensor_ops import common, convert
from zetta_utils.tensor_typing import TensorTypeVar

from .common import tiled_inference


@builder.register("BaseCoarsener")
@typechecked
@attrs.mutable
class BaseCoarsener:
    # Input int8 [   -127 .. 127] or uint8 [0 .. 255]
    # Output int8 Encodings [-127 .. 127]

    # Don't create the model during initialization for efficient serialization
    model_path: str
    abs_val_thr: float = 0.005
    ds_factor: int = 1
    output_channels: int = 1
    tile_pad_in: int = 0
    tile_size: int | None = None

    def __call__(
        self, src: TensorTypeVar, output_mask: None | TensorTypeVar = None
    ) -> TensorTypeVar:
        with torch.no_grad():
            device = "cuda" if torch.cuda.is_available() else "cpu"
            data_in = convert.to_torch(src, device=device)

            # load model during the call _with caching_
            model = convnet.utils.load_model(self.model_path, device=device, use_cache=True)

            # uint8 raw images or int8 encodings
            if data_in.dtype == torch.int8:
                data_in = data_in.float() / 127.0  # [-1.0 .. 1.0]
            elif data_in.dtype == torch.uint8:
                data_in = data_in.float() / 255.0  # [ 0.0 .. 1.0]
            else:
                raise ValueError(f"Unsupported src dtype: {data_in.dtype}")

            data_in = einops.rearrange(data_in, "C X Y Z -> Z C X Y")

            if self.tile_size is None:
                result = model(data_in)
            else:
                result = tiled_inference(
                    model,
                    data_in,
                    device=device,
                    tile_size=self.tile_size,
                    tile_pad_in=self.tile_pad_in,
                    ds_factor=self.ds_factor,
                    output_dtype=torch.float32,
                    output_channels=self.output_channels,
                )

            # Final layer assumed to be tanh
            assert result.abs().max() <= 1
            result[result.abs() < self.abs_val_thr] = 0

            if output_mask is not None:
                output_mask = einops.rearrange(output_mask, "C X Y Z -> Z C X Y")  # type: ignore
                result *= convert.to_torch(output_mask, device=result.device)

            result = (127.0 * result).round().clamp(-127, 127).type(torch.int8)
            result = volumetric.to_vol_layer_dtype(
                common.rearrange(result, "Z C X Y -> C X Y Z"),
            )
            return result
