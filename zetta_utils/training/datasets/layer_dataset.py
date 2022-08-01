from typing import Any

import attrs
import torch
import numpy as np
from typeguard import typechecked

import zetta_utils as zu
from zetta_utils.training.datasets.sample_indexers import SampleIndexer


def _convert_to_torch_nested(data):
    if isinstance(data, np.ndarray):
        result = zu.tensor.convert.to_torch(data)
    elif isinstance(data, dict):
        result = {k: _convert_to_torch_nested(v) for k, v in data.items()}
    else:
        result = data

    return result


@zu.builder.register("LayerDataset")
@typechecked
@attrs.frozen
class LayerDataset(torch.utils.data.Dataset):
    """Pytorch dataset wrapper around ``zu.io.Layer`` component.

    :param layer: Layer which will be used as a source of data.
    :param sample_indexer: Indexer which will be used to translate integer sample
        index to a corresponding index understood by the layer backend.

    """

    layer: zu.io.layer.Layer
    sample_indexer: SampleIndexer

    def __attrs_pre_init__(self):
        super().__init__()

    def __len__(self) -> int:
        return len(self.sample_indexer)

    def __getitem__(self, idx: int) -> Any:
        layer_idx = self.sample_indexer(idx)
        sample_raw = self.layer[layer_idx]
        sample = _convert_to_torch_nested(sample_raw)
        return sample
