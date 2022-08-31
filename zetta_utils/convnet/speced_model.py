from typing import Callable
import attrs
import torch


@attrs.frozen
class SpecedModel(torch.nn.Module):
    init_scheme: list[Callable]
    model: torch.nn.Module

    def __attrs_pre_init__(self):
        super().__init__()

    def __attrs_post_init__(self):
        for i in self.init_scheme:
            i(self.model)

    def forward(self, *args, **kwargs):
        self.model.forward(*args, **kwargs)
