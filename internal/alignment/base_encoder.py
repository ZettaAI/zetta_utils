import attrs
import einops
import torch
from typeguard import typechecked

from zetta_utils import builder, convnet


@builder.register("BaseEncoder")
@typechecked
@attrs.mutable
class BaseEncoder:
    # Input uint8 [   0 .. 255]
    # Output int8 Encodings [-127 .. 127]

    # Don't create the model during initialization for efficient serialization
    model_path: str
    abs_val_thr: float = 0.005
    uint_output: bool = False
    preserve_size: bool = False

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
            with torch.autocast(device_type=device):
                result = model(data_in.to(device))
            if self.preserve_size:
                x_cropped = data_in.shape[2] - result.shape[2]
                y_cropped = data_in.shape[3] - result.shape[3]
                if x_cropped % 2 != 0 or y_cropped % 2 != 0:
                    raise ValueError(
                        "`preserve_size` is unsupported for unevenly cropped outputs; received"
                        f" (in ZCXY) input shape: {data_in.shape}, output shape: {result.shape}"
                    )
                result = torch.nn.functional.pad(
                    result,
                    (x_cropped // 2, x_cropped // 2, y_cropped // 2, y_cropped // 2),
                    "constant",
                    0,
                )
            result = einops.rearrange(result, "Z C X Y -> C X Y Z")

            # Final layer assumed to be tanh
            assert result.abs().max() <= 1
            result[result.abs() < self.abs_val_thr] = 0
            if self.uint_output:
                # FOR LEGACY MODELS. to be removed
                result += 1
            result = 127.0 * result

        if self.uint_output:
            # FOR LEGACY MODELS. to be removed
            return result.type(torch.uint8)
        else:
            return result.round().type(torch.int8).clamp(-127, 127)
