import attrs
import einops
import torch
from typeguard import typechecked

from zetta_utils import builder, convnet


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

    def __call__(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        if (src != 0).sum() == 0 or (tgt != 0).sum() == 0:
            return torch.zeros_like(src[:1], dtype=torch.uint8)

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        # load model during the call _with caching_
        model = convnet.utils.load_model(self.model_path, device=device, use_cache=True)

        assert src.dtype == tgt.dtype
        src_zcxy = einops.rearrange(src, "C X Y Z -> Z C X Y").float()
        tgt_zcxy = einops.rearrange(tgt, "C X Y Z -> Z C X Y").float()

        if src.dtype == torch.uint8:
            data_in = torch.cat((src_zcxy, tgt_zcxy), 1) / 255.0
        elif src.dtype == torch.int8:
            data_in = torch.cat((src_zcxy, tgt_zcxy), 1) / 127.0

        with torch.no_grad():
            result = model(data_in.to(device))

        result = einops.rearrange(result, "Z C X Y -> C X Y Z")

        assert result.shape[0] == 1

        assert result.max() <= 1, "Final layer of misalignment detector assumed to be sigmoid"
        assert result.min() >= 0, "Final layer of misalignment detector assumed to be sigmoid"
        result = 255.0 * result

        return result.round().clamp(0.0, 255.0).byte().to(src.device)
