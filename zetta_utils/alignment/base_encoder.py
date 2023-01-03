import attrs
import einops
import torch
from typeguard import typechecked

from zetta_utils import builder, convnet


@builder.register("BaseEncoder")
@typechecked
@attrs.mutable
class BaseEncoder:
    # Don't create the model during initialization for efficient serialization
    model_path: str
    abs_val_thr: float = 0.005

    def __call__(self, src: torch.Tensor) -> torch.Tensor:
        if (src != 0).sum() == 0:
            result = src
        else:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

            # load model during the call _with caching_
            model = convnet.utils.load_model(self.model_path, device=device, use_cache=True)

            if src.dtype == torch.uint8:
                data_in = src.float() / 255.0
            elif src.dtype == torch.float32:
                data_in = src
            else:
                raise ValueError(f"Unsupported src dtype: {src.dtype}")

            data_in = einops.rearrange(data_in, "C X Y Z -> Z C X Y")
            result = model(data_in.to(device))
            result = einops.rearrange(result, "Z C X Y -> C X Y Z")

            # Final layer assumed to be tanh
            assert result.abs().max() <= 1
            result[result.abs() < self.abs_val_thr] = 0
            result = 255.0 * (result + 1.0) / 2
            result = result.byte()

        return result
