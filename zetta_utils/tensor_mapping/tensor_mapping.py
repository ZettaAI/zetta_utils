from __future__ import annotations

import collections

import attrs
import torch
from typeguard import typechecked


@typechecked
@attrs.mutable
class TensorMapping(collections.abc.MutableMapping[str, torch.Tensor]):
    tensors: dict[str, torch.Tensor]

    def __len__(self):
        return len(self.tensors)

    def __delitem__(self, name):
        del self.tensors[name]

    def __iter__(self):
        return iter(self.tensors)

    def __getitem__(self, name) -> torch.Tensor:
        return self.tensors[name]

    def __setitem__(self, name, data: torch.Tensor):
        self.tensors[name] = data

    def __add__(self, other: TensorMapping | torch.Tensor) -> TensorMapping:
        result = TensorMapping({k: v.clone() for k, v in self.tensors.items()})
        if isinstance(other, TensorMapping):
            for k in self.tensors.keys():
                result.tensors[k] += other[k]
        else:
            for k in self.tensors.keys():
                result.tensors[k] += other
        return result

    def __mul__(self, other: TensorMapping | torch.Tensor) -> TensorMapping:
        result = TensorMapping({k: v.clone() for k, v in self.tensors.items()})
        if isinstance(other, TensorMapping):
            for k in result.tensors.keys():
                result.tensors[k] *= other[k]
        else:
            for k in self.tensors.keys():
                result.tensors[k] *= other

        return result
