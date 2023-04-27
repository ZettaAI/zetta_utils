import attrs
import einops
import torch
from typeguard import typechecked

from zetta_utils import builder, convnet


@builder.register("BaseCoarsenerSubchunkable")
@typechecked
@attrs.mutable
class BaseCoarsenerSubchunkable:
    # Input int8 [   -127 .. 127] or uint8 [0 .. 255]
    # Output int8 Encodings [-127 .. 127]

    # Don't create the model during initialization for efficient serialization
    model_path: str
    abs_val_thr: float = 0.005
    ds_factor: int = 1

    def __call__(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
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
            result = torch.zeros_like(
                data_in[
                    ...,
                    : data_in.shape[-2] // self.ds_factor,
                    : data_in.shape[-1] // self.ds_factor,
                ]
            ).float()

            if (data_in != 0).sum() > 0.0:
                result = model(data_in)


            result = einops.rearrange(result, "Z C X Y -> C X Y Z")

            # Final layer assumed to be tanh
            assert result.abs().max() <= 1
            result[result.abs() < self.abs_val_thr] = 0
            result = 127.0 * result

            return result.round().type(torch.int8).clamp(-127, 127)
