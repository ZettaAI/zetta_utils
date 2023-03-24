# pylint: disable = useless-super-delegation, no-self-use, redefined-builtin
from __future__ import annotations

from collections.abc import Sequence as AbcSequence
from functools import partial
from types import BuiltinMethodType
from typing import Any, Callable, List, Literal, Sequence

import torch
import torch.nn.functional as F
from typeguard import typechecked

from zetta_utils import builder

NN_MANUAL = [
    "Sequential",
    "Upsample",
]

for k in dir(torch.nn):
    if k not in NN_MANUAL and k[0].isupper():
        builder.register(f"torch.nn.{k}")(getattr(torch.nn, k))

for module in [torch, torch.nn.functional, torch.linalg, torch.fft]:
    for k in dir(module):
        attr = getattr(module, k)
        if isinstance(attr, BuiltinMethodType) and not k.startswith("_"):
            builder.register(f"{module.__name__}.{k}")(attr)


@builder.register("torch.nn.Sequential")
@typechecked
def sequential_builder(modules: List[Any]) -> torch.nn.Sequential:  # pragma: no cover
    return torch.nn.Sequential(*modules)


@builder.register("torch.nn.Upsample")
@typechecked
def upsample_builder(
    size: None | int | Sequence[int] = None,
    scale_factor: None | float | Sequence[float] = None,
    mode: Literal["nearest", "linear", "bilinear", "bicubic", "trilinear"] = "nearest",
    align_corners: bool | None = False,
    recompute_scale_factor: bool | None = None,
) -> torch.nn.Upsample:  # pragma: no cover
    if isinstance(scale_factor, AbcSequence):
        scale_factor = tuple(scale_factor)
    if isinstance(size, AbcSequence):
        size = tuple(size)
    return torch.nn.Upsample(
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
    )


# need num_channels to go first to be compatible with batchnorm
@builder.register("build_group_norm")
def build_group_norm(
    num_channels, num_groups, eps=1e-05, affine=True
) -> torch.nn.GroupNorm:  # pragma: no cover
    return torch.nn.GroupNorm(num_groups, num_channels, eps, affine)


@builder.register("SplitTuple")
class SplitTuple(torch.nn.Module):  # pragma: no cover
    def __init__(self):
        super().__init__()

    def forward(self, x):
        half = x.shape[0] // 2
        return torch.cat([x[0:half, ...], x[half:, ...]], 1)


@builder.register("RescaleValues")
class RescaleValues(torch.nn.Module):  # pragma: no cover
    def __init__(self, in_range, out_range):
        super().__init__()
        self.in_min = in_range[0]
        self.in_max = in_range[1]
        self.out_min = out_range[0]
        self.out_max = out_range[1]

    def forward(self, x):
        x = (x - self.in_min) / (self.in_max - self.in_min)
        x = x * (self.out_max - self.out_min) + self.out_min
        return x


@builder.register("View")
class View(torch.nn.Module):  # pragma: no cover
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


@builder.register("Flatten")
class Flatten(torch.nn.Module):  # pragma: no cover
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


@builder.register("Unflatten")
class Unflatten(torch.nn.Module):  # pragma: no cover
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1, 1, 1)


@builder.register("MaxPool2DFlatten")
class MaxPool2DFlatten(torch.nn.Module):  # pragma: no cover
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert x.shape[-1] == x.shape[-2]
        batch_size = x.shape[0]
        x = F.max_pool2d(x, x.shape[-1])
        x = x.squeeze()
        if batch_size == 1:
            x = x.unsqueeze(0)
        return x


@builder.register("AvgPool2DFlatten")
class AvgPool2DFlatten(torch.nn.Module):  # pragma: no cover
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert x.shape[-1] == x.shape[-2]
        batch_size = x.shape[0]
        x = F.avg_pool2d(x, x.shape[-1])
        x = x.squeeze()
        if batch_size == 1:
            x = x.unsqueeze(0)
        return x


@builder.register("Clamp")
class Clamp(torch.nn.Module):  # pragma: no cover
    def __init__(self, min, max):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return torch.clamp(x, self.min, self.max)


@builder.register("UpConv")
class UpConv(torch.nn.Module):  # pragma: no cover
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        upsampler: Callable[..., torch.nn.Module] = partial(torch.nn.Upsample, scale_factor=2),
        conv: Callable[..., torch.nn.Module] = partial(torch.nn.Conv2d, padding="same"),
    ):
        super().__init__()
        self.upsampler = upsampler()
        self.conv = conv(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size
        )

    def forward(self, x):
        x = self.upsampler(x)
        return self.conv(x)
