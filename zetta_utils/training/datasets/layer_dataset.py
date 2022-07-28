from typing import Any

import attrs
import torch
import zetta_utils as zu

from zetta_utils.training.datasets.sample_indexers import SampleIndexer


@zu.spec_parser.register("LayerDataset")
@attrs.frozen
class LayerDataset(torch.utils.data.Dataset):
    layer: zu.io.layers.Layer
    sample_indexer: SampleIndexer

    def __attrs_pre_init__(self):
        super().__init__()

    def __len__(self) -> int:
        return len(self.sample_indexer)

    def __getitem__(self, idx: int) -> Any:
        layer_idx = self.sample_indexer(idx)
        sample = self.layer[layer_idx]
        return sample
