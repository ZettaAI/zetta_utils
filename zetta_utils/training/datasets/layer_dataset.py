from typing import Any

import attrs
import numpy as np
import torch
from typeguard import typechecked

from zetta_utils import builder, tensor_ops
from zetta_utils.layer import Layer

from .sample_indexers import SampleIndexer


def _convert_to_torch_nested(data):
    if isinstance(data, np.ndarray):
        result = tensor_ops.convert.to_torch(data)
    elif isinstance(data, dict):
        result = {k: _convert_to_torch_nested(v) for k, v in data.items()}  # type: ignore
    else:
        result = data

    return result


@builder.register("LayerDataset")
@typechecked
@attrs.frozen
class LayerDataset(torch.utils.data.Dataset):
    """PyTorch dataset wrapper around `zetta_utils.layer.Layer` component.

    :param layer: Layer which will be used as a source of data.
    :param sample_indexer: Indexer which will be used to translate integer sample
        index to a corresponding index understood by the layer backend.
    :param length_override: Overide `len()` to return a specific value. Useful
        when used in a JointDataset to weigh `LayerDataset`s differently.
        Caution: typically to be useful the `sample_indexer` must return random
        sample when given the same index (i.e., use RandomIndexer).
    """

    layer: Layer
    sample_indexer: SampleIndexer
    length_override: int | None = None

    def __attrs_pre_init__(self):
        super().__init__()

    def __len__(self) -> int:
        if self.length_override is not None:
            return self.length_override
        return len(self.sample_indexer)

    def __getitem__(self, idx: int) -> Any:
        layer_idx = self.sample_indexer(idx)
        sample_raw = self.layer.read_with_procs(layer_idx)
        sample = _convert_to_torch_nested(sample_raw)
        return sample
