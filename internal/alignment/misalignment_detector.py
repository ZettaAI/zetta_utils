import attrs
import einops
import torch
from typeguard import typechecked

from zetta_utils import builder, convnet
from zetta_utils.tensor_ops import convert
from zetta_utils.tensor_typing import TensorTypeVar


@builder.register("naive_misd")
def naive_misd(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    result = ((src == 0) + (tgt == 0)).byte()
    return result


@builder.register("MisalignmentDetector")
@typechecked
@attrs.mutable
class MisalignmentDetector:
    # Don't create the model during initialization for efficient serialization
    model_path: str
    apply_sigmoid: bool = True

    def __call__(self, src: TensorTypeVar, tgt: TensorTypeVar) -> torch.Tensor:
        if not src.any() or not tgt.any():
            return torch.zeros_like(convert.to_torch(src[:1]), dtype=torch.uint8)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        src_in = convert.to_torch(src, device=device)
        tgt_in = convert.to_torch(tgt, device=device)

        # load model during the call _with caching_
        model = convnet.utils.load_model(self.model_path, device=device, use_cache=True)

        assert src_in.dtype == tgt_in.dtype
        src_zcxy = einops.rearrange(src_in, "C X Y Z -> Z C X Y").float()
        tgt_zcxy = einops.rearrange(tgt_in, "C X Y Z -> Z C X Y").float()

        if src_in.dtype == torch.uint8:
            data_in = torch.cat((src_zcxy, tgt_zcxy), 1) / 255.0
        elif src_in.dtype == torch.int8:
            data_in = torch.cat((src_zcxy, tgt_zcxy), 1) / 127.0

        with torch.no_grad():
            with torch.autocast(device_type=device):
                result = model(data_in.to(device))
                if self.apply_sigmoid:
                    result.sigmoid_()

        result = einops.rearrange(result, "Z C X Y -> C X Y Z")

        assert result.shape[0] == 1

        assert result.max() <= 1
        assert result.min() >= 0
        result = 255.0 * result

        return result.round().clamp(0.0, 255.0).byte().to(src_in.device)
