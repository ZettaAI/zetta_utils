import torch

from zetta_utils import builder


@builder.register("detect_consecutive_masks")
def detect_consecutive_masks(
    src: torch.Tensor,
    num_consecutive: int = 3,
) -> torch.Tensor:
    """
    Identify locations in `src` that participate in `num_consecutive` sections
    of a mask. Using `num_consecutive=3`, this is identical to the boolean function:

    `(z-2 & z-1 & z) | (z-1 & z & z+1) | (z & z+1 & z+2)`

    :param src: tensor
    :param num_consecutive: number z sections that a location must
        be masked to be considered part of the consecutive mask

    """
    if src.shape[-1] < num_consecutive:
        raise ValueError(
            f"Expected the number of masks to be >={num_consecutive}, but got {src.shape[-1]}"
        )
    # sum over z with kernel of size num_consecutive
    # pylint: disable=invalid-name
    c, x, y, z = src.shape
    reshaped_src = src.reshape(x * y, c, z)
    kernel = torch.ones((1, 1, num_consecutive), dtype=src.dtype)
    # use full padding to help with propagation later
    src_sum = torch.nn.functional.conv1d(reshaped_src, kernel, padding=num_consecutive - 1)
    is_consecutive = src_sum >= num_consecutive
    # propagate is_consecutive mask to all participating sections
    # only needs to participate in one consecutive range to be masked
    participates = torch.logical_or(
        *(is_consecutive[:, :, i : z + i] for i in range(num_consecutive))
    )
    return participates.reshape(c, x, y, z).to(src.dtype)
