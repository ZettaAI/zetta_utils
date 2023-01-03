import einops
import torch

from zetta_utils import builder


@builder.register("apply_base_encoder")
def apply_base_encoder(
    src: torch.Tensor,
    model_spec_path: str,
):
    data_proc = einops.rearrange(src.clone(), "C X Y Z -> Z C X Y").float()

    if (src != 0).sum() > 0:
        model = builder.build(path=model_spec_path, use_cache=True)
        if data_proc.min() >= 0 and data_proc.max() > 1:
            # assuming uint8 0-255 range
            data_proc /= 255

        result = model(data_proc)
    else:
        result = data_proc

    assert result.abs().max() <= 1

    result[result.abs() < 0.005] = 0
    result += 1.0
    result /= 2
    result *= 255.0
    result = einops.rearrange(result, "Z C X Y -> C X Y Z").byte()
    return result
