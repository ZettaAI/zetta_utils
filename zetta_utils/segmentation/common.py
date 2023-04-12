from __future__ import annotations

import attrs
import torch
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.layer import DataProcessor


@builder.register("MultiHeadedProcessor")
@typechecked
@attrs.frozen
class MultiHeadedProcessor(DataProcessor):  # pragma: no cover
    spec: dict[str, list[str]]

    def __call__(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for source, targets in self.spec.items():
            for target in targets:
                data[target] = data[source]
                if source + "_mask" in data:
                    data[target + "_mask"] = data[source + "_mask"]
        return data
