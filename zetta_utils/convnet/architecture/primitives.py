# pylint: disable = useless-super-delegation, no-self-use, redefined-builtin
from typing import Any, List

import torch
import torch.nn.functional as F
from typeguard import typechecked

from zetta_utils import builder


# Wrapper for building a Sequential
@builder.register("Sequential")
@typechecked
def sequential_builder(modules: List[Any]) -> torch.nn.Sequential:  # pragma: no cover
    return torch.nn.Sequential(*modules)


# Activation
builder.register("LeakyReLU")(torch.nn.LeakyReLU)
builder.register("ReLU")(torch.nn.ReLU)
builder.register("ELU")(torch.nn.ELU)
builder.register("Tanh")(torch.nn.ReLU)
builder.register("Sigmoid")(torch.nn.Sigmoid)
builder.register("Hardsigmoid")(torch.nn.Hardsigmoid)
builder.register("LogSigmoid")(torch.nn.LogSigmoid)
builder.register("LogSoftmax")(torch.nn.LogSoftmax)

# Normalization
builder.register("BatchNorm2d")(torch.nn.BatchNorm2d)
builder.register("BatchNorm3d")(torch.nn.BatchNorm3d)
builder.register("InstanceNorm2d")(torch.nn.InstanceNorm2d)
builder.register("InstanceNorm3d")(torch.nn.InstanceNorm3d)
# need num_channels to go first to be compatible with batchnorm
@builder.register("GroupNorm")
def compatible_group_norm(
    num_channels, num_groups, eps=1e-05, affine=True
) -> torch.nn.GroupNorm:  # pragma: no cover
    return torch.nn.GroupNorm(num_groups, num_channels, eps, affine)


# Convolutions
builder.register("Conv2d")(torch.nn.Conv2d)
builder.register("Conv3d")(torch.nn.Conv3d)
builder.register("ConvTranspose2d")(torch.nn.ConvTranspose2d)
builder.register("ConvTranspose3d")(torch.nn.ConvTranspose3d)

# Interpolation
# Can be done via zu.tensor_ops.interpolate

# Primitives from modules
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
