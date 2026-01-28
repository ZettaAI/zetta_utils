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
    :param include_index_metadata: If True, include "_idx_resolution" and "_idx_start"
        in the output dict for coordinate conversion.

    """

    layer: Layer
    sample_indexer: SampleIndexer
    include_index_metadata: bool = False

    def __attrs_pre_init__(self):
        super().__init__()

    def __len__(self) -> int:
        return len(self.sample_indexer)

    def __getitem__(self, idx: int) -> Any:
        layer_idx = self.sample_indexer(idx)
        sample_raw = self.layer.read_with_procs(layer_idx)
        sample = _convert_to_torch_nested(sample_raw)
        if self.include_index_metadata and isinstance(sample, dict):
            # Include index metadata for coordinate conversion
            # resolution: voxel size in nm (Vec3D)
            # start: voxel coordinates of chunk origin (Vec3D[int])
            # Store as tensors so collation stacks them into [batch, 3]
            sample["_idx_resolution"] = torch.tensor(
                list(layer_idx.resolution), dtype=torch.float32
            )
            sample["_idx_start"] = torch.tensor(list(layer_idx.start), dtype=torch.int64)
        return sample
