from typing import Tuple

import einops
import torch
import torchfields  # pylint: disable=unused-import


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


def invert_field(field: torch.Tensor, num_iter: int = 200, lr: float = 1e-5):
    field_zcxy = einops.rearrange(field, "C X Y Z -> Z C X Y").field().cuda()  # type: ignore
    inverse_zcxy = (-field_zcxy).clone().field().from_pixels()
    inverse_zcxy.requires_grad = True

    optimizer = torch.optim.Adam([inverse_zcxy], lr=lr)
    for _ in range(num_iter):
        loss = inverse_zcxy(field_zcxy.from_pixels()).pixels().abs().sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return einops.rearrange(inverse_zcxy.pixels(), "Z C X Y -> C X Y Z")
