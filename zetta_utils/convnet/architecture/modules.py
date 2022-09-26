import torch
import torch.nn as nn
import torch.nn.functional as F


class SplitTuple(nn.Module):  # pragma: no cover
    def __init__(self):
        super().__init__()

    def forward(self, x):
        half = x.shape[0] // 2
        return torch.cat([x[0:half, ...], x[half:, ...]], 1)


class RescaleValues(nn.Module):  # pragma: no cover
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


class View(nn.Module):  # pragma: no cover
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.view(*self.args)


class Flatten(nn.Module):  # pragma: no cover
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class Unflatten(nn.Module):  # pragma: no cover
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1, 1, 1)


class MaxPoolFlatten(nn.Module):  # pragma: no cover
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.max_pool2d(x, x.shape[-1])


class AvgPoolFlatten(nn.Module):  # pragma: no cover
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.avg_pool2d(x, x.shape[-1])


class Clamp(nn.Module):  # pragma: no cover
    def __init__(self, minimum, maximum):
        super().__init__()
        self.minimum = minimum
        self.maximum = maximum

    def forward(self, x):
        return torch.clamp(x, self.minimum, self.maximum)
