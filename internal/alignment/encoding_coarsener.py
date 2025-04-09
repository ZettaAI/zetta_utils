import attrs
import einops
import torch
from typeguard import typechecked

from zetta_utils import builder, convnet


@builder.register("EncodingCoarsener")
@typechecked
@attrs.mutable
class EncodingCoarsener:
    # Input int8 [   -127 .. 127]
    # Output int8 Encodings [-127 .. 127]

    # Don't create the model during initialization for efficient serialization
    model_path: str
    abs_val_thr: float = 0.005

    def __call__(self, src: torch.Tensor) -> torch.Tensor:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        # load model during the call _with caching_
        model = convnet.utils.load_model(self.model_path, device=device, use_cache=True)
        if src.dtype == torch.int8:
            data_in = src.float() / 127.0
        else:
            raise ValueError(f"Unsupported src dtype: {src.dtype}")

        data_in = einops.rearrange(data_in, "C X Y Z -> Z C X Y")
        result = model(data_in.to(device))
        result = einops.rearrange(result, "Z C X Y -> C X Y Z")

        # Final layer assumed to be tanh
        assert result.abs().max() <= 1
        result[result.abs() < self.abs_val_thr] = 0
        result = 127.0 * (result)
        return result.round().type(torch.int8).clamp(-127, 127)
