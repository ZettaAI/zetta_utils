from typing import Literal, Tuple

import einops
import torch
import torchfields  # pylint: disable=unused-import
from torch.optim.lr_scheduler import ReduceLROnPlateau

from zetta_utils import builder
from zetta_utils.augmentations.tensor import rand_perlin_2d_octaves


def profile_field2d_percentile(
    field: torch.Tensor,  # C, X, Y, Z
    high: float = 25,
    low: float = 75,
) -> Tuple[int, int]:

    nonzero_field_mask = (field[0] != 0) & (field[1] != 0)

    nonzero_field = field[..., nonzero_field_mask].squeeze()

    if nonzero_field.sum() == 0 or len(nonzero_field.shape) == 1:
        result = (0, 0)
    else:
        low_l = percentile(nonzero_field, low)
        high_l = percentile(nonzero_field, high)
        mid = 0.5 * (low_l + high_l)
        result = (int(mid[0]), int(mid[1]))

    return result


def percentile(field: torch.Tensor, q: float):
    # https://gist.github.com/spezold/42a451682422beb42bc43ad0c0967a30
    """
    Return the ``q``-th percentile of the flattened input tensor's data.
    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.
    :param field: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(0.01 * float(q) * (field.shape[1] - 1))
    result = field.kthvalue(k, dim=1).values
    return result


@builder.register("invert_field")
def invert_field(src: torch.Tensor, mode: Literal["opti", "torchfields"] = "opti") -> torch.Tensor:
    if src.abs().sum() == 0:
        return src

    if mode == "opti":
        result = invert_field_opti(src)
    else:
        src_zcxy = einops.rearrange(src, "C X Y Z -> Z C X Y").cuda().field_()  # type: ignore
        with torchfields.set_identity_mapping_cache(True):
            result_zcxy = (~(src_zcxy.from_pixels())).pixels()
        result = einops.rearrange(result_zcxy, "Z C X Y -> C X Y Z")
    return result


def invert_field_opti(src: torch.Tensor, num_iter: int = 200, lr: float = 1e-3) -> torch.Tensor:
    if src.abs().sum() == 0:
        return src

    src_zcxy = (
        einops.rearrange(src, "C X Y Z -> Z C X Y").cuda().field_().from_pixels()  # type: ignore
    )

    with torchfields.set_identity_mapping_cache(True):
        inverse_zcxy = (-src_zcxy).clone()
        inverse_zcxy.requires_grad = True

        optimizer = torch.optim.Adam([inverse_zcxy], lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, "min", patience=5, min_lr=1e-5)
        for _ in range(num_iter):
            loss = inverse_zcxy(src_zcxy).pixels().abs().sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

        return einops.rearrange(inverse_zcxy.detach().pixels(), "Z C X Y -> C X Y Z")


@builder.register("gen_biased_perlin_noise_field")
def gen_biased_perlin_noise_field(
    shape,
    *,
    res,
    octaves=1,
    persistence=0.5,
    field_magn_thr_px=1.0,
    max_displacement_px=None,
    device="cuda",
):
    """Generates a perlin noise vector field with the provided median and maximum vector length."""
    eps = 1e-7
    perlin = rand_perlin_2d_octaves(shape, res, octaves, persistence, device=device)
    warp_field = einops.rearrange(perlin, "C X Y Z -> Z C X Y").field_()

    vec_length = warp_field.norm(dim=1, keepdim=True).tensor_()
    vec_length_median = torch.median(vec_length)
    vec_length_centered = vec_length - vec_length_median

    vec_length_target = torch.where(
        vec_length_centered < 0,
        vec_length_centered * field_magn_thr_px / abs(vec_length_centered.min())
        + field_magn_thr_px,
        vec_length_centered
        * (max_displacement_px - field_magn_thr_px)
        / abs(vec_length_centered.max())
        + field_magn_thr_px,
    )

    warp_field *= vec_length_target / (vec_length + eps)
    return einops.rearrange(warp_field, "Z C X Y -> C X Y Z").tensor_()
