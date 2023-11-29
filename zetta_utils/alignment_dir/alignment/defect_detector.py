import attrs
import einops
import torch
from typeguard import typechecked

from zetta_utils import builder, convnet


@builder.register("DefectDetector")
@typechecked
@attrs.mutable
class DefectDetector:
    # Input uint8 [   0 .. 255]
    # Output uint8 Prediction [0 .. 255]

    # Don't create the model during initialization for efficient serialization
    model_path: str
    tile_pad_in: int = 32
    tile_size: int = 448

    def __call__(self, src: torch.Tensor) -> torch.Tensor:
        if (src != 0).sum() == 0:
            result = torch.zeros_like(src).float()
        else:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

            # load model during the call _with caching_
            model = convnet.utils.load_model(self.model_path, device=device, use_cache=True)

            if src.dtype == torch.uint8:
                data_in = src.float() / 255.0  # [0.0 .. 1.0]
            else:
                raise ValueError(f"Unsupported src dtype: {src.dtype}")

            data_in = einops.rearrange(data_in, "C X Y Z -> Z C X Y")
            data_in = data_in.to(device=device)
            with torch.no_grad():
                result = torch.zeros_like(
                    data_in[
                        ...,
                        : data_in.shape[-2],
                        : data_in.shape[-1],
                    ]
                ).float()

                tile_pad_out = self.tile_pad_in

                for x in range(
                    self.tile_pad_in, data_in.shape[-2] - self.tile_pad_in, self.tile_size
                ):
                    x_start = x - self.tile_pad_in
                    x_end = x + self.tile_size + self.tile_pad_in
                    for y in range(
                        self.tile_pad_in, data_in.shape[-1] - self.tile_pad_in, self.tile_size
                    ):
                        y_start = y - self.tile_pad_in
                        y_end = y + self.tile_size + self.tile_pad_in
                        tile = data_in[:, :, x_start:x_end, y_start:y_end]
                        if (tile != 0).sum() > 0.0:
                            tile_result = model(tile)
                            if tile_pad_out > 0:
                                tile_result = tile_result[
                                    :, :, tile_pad_out:-tile_pad_out, tile_pad_out:-tile_pad_out
                                ]

                            result[
                                :,
                                :,
                                x : x + tile_result.shape[-2],
                                y : y + tile_result.shape[-1],
                            ] = tile_result

            result = einops.rearrange(result, "Z C X Y -> C X Y Z")
            result = 255.0 * torch.sigmoid(result)
            result[src == 0.0] = 0.0

        return result.round().clamp(0, 255).type(torch.uint8)
