import torch
import einops
from typeguard import typechecked

from zetta_utils import builder


@builder.register("invert_field")
@typechecked
def invert_field(field: torch.Tensor, in_pixels: bool = True):
    # C X Y Z
    assert in_pixels

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    field_zcxy = einops.rearrange(field, "C X Y Z -> Z C X Y").field().to(device)

    result_zcxy = ~(field_zcxy.from_pixels()).pixels()
    result = einops.rearrange(result_zcxy, "Z C X Y -> C X Y Z").to(field.device)
    return result
