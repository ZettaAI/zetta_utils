from __future__ import annotations

import attrs
import cc3d
import numpy as np
import torch
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.layer import DataProcessor
from zetta_utils.tensor_ops import convert


@builder.register("EmbeddingProcessor")
@typechecked
@attrs.mutable
class EmbeddingProcessor(DataProcessor):  # pragma: no cover
    source: str
    target: str
    split_label: bool = False

    def __call__(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

        # Segmentation and mask
        seg = data[self.source]
        msk = data[self.source + "_mask"]

        # Target
        data[self.target] = seg
        data[self.target + "_mask"] = msk

        # Split label
        if self.split_label:
            seg_np = convert.to_np(seg).astype(np.uint64)
            seg_cc = cc3d.connected_components(seg_np)
            seg_split = convert.astype(seg_cc, seg, cast=True)
            data[self.target + "_split"] = seg_split

        return data
