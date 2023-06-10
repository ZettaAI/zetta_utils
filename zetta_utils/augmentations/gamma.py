from __future__ import annotations

from typing import Literal, Sequence

import attrs
import torch
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.layer import JointIndexDataProcessor
from zetta_utils.layer.volumetric import VolumetricIndex


@typechecked
def adjust_gamma(data: torch.Tensor, gamma: float, gain: float) -> torch.Tensor:
    return gain * (data ** gamma)


@builder.register("GammaAugment")
@typechecked
@attrs.frozen
class GammaAugment(JointIndexDataProcessor):
    key: str
    z_section_wise: bool

    log2_gamma: float | Sequence[float] = 1
    gain: int = 1

    def process_index(
        self, idx: VolumetricIndex, mode: Literal["read", "write"]
    ) -> VolumetricIndex:
        return idx

    def process_data(
        self, data: dict[str, torch.Tensor], mode: Literal["read", "write"]
    ) -> dict[str, torch.Tensor]:
        raw = data[self.key]
        assert raw.min().item() >= 0 and raw.max().item() <= 1

        # TODO: use distributions to choose gamma randomly
        # gamma = distribution
        gamma = 1

        if self.z_section_wise:
            for z in range(raw.shape[-1]):
                data[self.key][..., z] = adjust_gamma(raw[..., z], gamma, self.gain)
        else:
            data[self.key] = adjust_gamma(raw, gamma, self.gain)
        return data
