from __future__ import annotations

from functools import partial

import numpy as np
import torch
from torch import nn
from typeguard import typechecked

from zetta_utils import builder


@builder.register("BinaryClassBalancer")
@typechecked
class BinaryClassBalancer(nn.Module):  # pragma: no cover
    """
    Computes a weight map by balancing foreground/background.

    :param weight0:
    :param weight1:
    :param clipmin:
    :param clipmax:
    :param group:
    """

    def __init__(
        self,
        weight0: float | None = None,
        weight1: float | None = None,
        clipmin: float = 0.01,
        clipmax: float = 0.99,
        group: int = 0,
    ):
        super().__init__()
        assert weight0 > 0 if weight0 is not None else True
        assert weight1 > 0 if weight1 is not None else True
        self.weight0 = weight0
        self.weight1 = weight1
        self.dynamic = (weight0 is None) and (weight1 is None)
        self.clip = partial(np.clip, a_min=clipmin, a_max=clipmax)
        self.group = group

    def forward(
        self,
        trgt: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(trgt, dtype=torch.float32)

        num_channels = trgt.shape[-4]
        group = self.group if self.group > 0 else num_channels

        balanced = []
        for i in range(0, num_channels, group):
            start = i
            end = min(i + group, num_channels)
            balanced.append(
                self._balance(
                    trgt[..., start:end, :, :, :],
                    mask[..., start:end, :, :, :],
                )
            )
        result = torch.cat(balanced, dim=-4)
        return result

    def _balance(self, trgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        dtype = mask.dtype
        ones = mask * torch.eq(trgt, 1).type(dtype)
        zeros = mask * torch.eq(trgt, 0).type(dtype)

        # Dynamic balancing
        if self.dynamic:

            n_ones = ones.sum().item()
            n_zeros = zeros.sum().item()
            if (n_ones + n_zeros) > 0:
                ones *= self.clip(n_zeros / (n_ones + n_zeros))
                zeros *= self.clip(n_ones / (n_ones + n_zeros))

        # Static weighting
        else:

            if self.weight1 is not None:
                ones *= self.weight1

            if self.weight0 is not None:
                zeros *= self.weight0

        return (ones + zeros).type(dtype)
