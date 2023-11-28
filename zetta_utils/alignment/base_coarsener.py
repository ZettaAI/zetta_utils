import attrs
import einops
import torch
from typeguard import typechecked

from zetta_utils import builder, convnet


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
    tile_pad_in: int = 128
    tile_size: int = 1024

    def __call__(self, src: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # load model during the call _with caching_
            model = convnet.utils.load_model(self.model_path, device=device, use_cache=True)

            # uint8 raw images or int8 encodings
            if src.dtype == torch.int8:
                data_in = src.float() / 127.0  # [-1.0 .. 1.0]
            elif src.dtype == torch.uint8:
                data_in = src.float() / 255.0  # [ 0.0 .. 1.0]
            else:
                raise ValueError(f"Unsupported src dtype: {src.dtype}")

            data_in = einops.rearrange(data_in, "C X Y Z -> Z C X Y").to(device)
            result = torch.zeros(
                data_in.shape[0],
                self.output_channels,
                data_in.shape[-2] // self.ds_factor,
                data_in.shape[-1] // self.ds_factor,
                dtype=torch.float32,
                layout=data_in.layout,
                device=data_in.device
            )
            tile_pad_out = self.tile_pad_in // self.ds_factor

            for x in range(self.tile_pad_in, data_in.shape[-2] - self.tile_pad_in, self.tile_size):
                x_start = x - self.tile_pad_in
                x_end = x + self.tile_size + self.tile_pad_in
                for y in range(
                    self.tile_pad_in, data_in.shape[-1] - self.tile_pad_in, self.tile_size
                ):
                    y_start = y - self.tile_pad_in
                    y_end = y + self.tile_size + self.tile_pad_in
                    tile = data_in[:, :, x_start:x_end, y_start:y_end]
                    if (tile != 0).sum() > 0.0:
                        with torch.autocast(device_type=device):
                            tile_result = model(tile)
                        if tile_pad_out > 0:
                            tile_result = tile_result[
                                :, :, tile_pad_out:-tile_pad_out, tile_pad_out:-tile_pad_out
                            ]

                        result[
                            :,
                            :,
                            x // self.ds_factor : x // self.ds_factor + tile_result.shape[-2],
                            y // self.ds_factor : y // self.ds_factor + tile_result.shape[-1],
                        ] = tile_result

            result = einops.rearrange(result, "Z C X Y -> C X Y Z")

            # Final layer assumed to be tanh
            assert result.abs().max() <= 1
            result[result.abs() < self.abs_val_thr] = 0
            result = 127.0 * result

            return result.round().type(torch.int8).clamp(-127, 127)
