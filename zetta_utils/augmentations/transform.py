from __future__ import annotations

import math
from itertools import product
from typing import Protocol, Sequence, TypeVar

import attrs
import numpy as np
import scipy
import torch
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.distributions import Distribution, to_distribution, uniform_distr
from zetta_utils.tensor_ops import convert
from zetta_utils.tensor_typing import TensorTypeVar

T = TypeVar("T")


class DataTransform(Protocol[T]):
    def __call__(self, __data: T) -> T:
        ...

    def prepare(self) -> None:
        ...


@builder.register("RandomFill")
@typechecked
@attrs.mutable
class RandomFill(DataTransform):
    fill: float | Distribution = 0
    frozen: bool = False

    fill_distr: Distribution = attrs.field(init=False)

    prepared_fill: float | None = attrs.field(init=False, default=None)

    def __attrs_post_init__(self):
        self.fill_distr = to_distribution(self.fill)

    def prepare(self) -> None:
        self.prepared_fill = self.fill_distr()

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if self.frozen:
            assert self.prepared_fill is not None
            value = self.prepared_fill
        else:
            value = self.fill_distr()
        return data.fill_(value)


@typechecked
@attrs.mutable
class PartialSection(DataTransform):
    transform: DataTransform

    frozen: bool = False
    per_partial: bool = False

    distr_x: Distribution = attrs.field(init=False)
    distr_y: Distribution = attrs.field(init=False)

    prepared_x: float | None = attrs.field(init=False, default=None)
    prepared_y: float | None = attrs.field(init=False, default=None)
    prepared_coins: Sequence[float] | None = attrs.field(init=False, default=None)

    def __attrs_post_init__(self) -> None:
        self.distr_x: Distribution = uniform_distr()
        self.distr_y: Distribution = uniform_distr()

    def prepare(self) -> None:
        self.transform.prepare()
        self.__prepare_pivot()
        self.prepared_coins = np.random.uniform(0, 1, 4).tolist()

    def __prepare_pivot(self) -> None:
        self.prepared_x = self.distr_x()
        self.prepared_y = self.distr_y()

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        # If not frozen, prepare random params for every transform
        if not self.frozen:
            self.prepare()

        # Pick a pivot point
        assert self.prepared_x is not None
        assert self.prepared_y is not None
        pivot_x = int(math.floor(self.prepared_x * data.shape[-2]))
        pivot_y = int(math.floor(self.prepared_y * data.shape[-1]))

        # Process each quadrant independently
        assert self.prepared_coins is not None
        quads = list(
            product(
                [slice(None, pivot_x), slice(pivot_x, None)],
                [slice(None, pivot_y), slice(pivot_y, None)],
            )
        )
        # If not per_partial, apply transform to entire data.
        transformed_data = self.transform(data) if not self.per_partial else None

        for coin, quad in zip(self.prepared_coins, quads):
            if coin < 0.5:
                if self.per_partial:
                    self.transform.prepare()
                    result = self.transform(data[..., quad[0], quad[1]])
                else:
                    result = transformed_data[..., quad[0], quad[1]]
                data[..., quad[0], quad[1]] = result

        return data


@typechecked
def apply_gaussian_filter(
    data: TensorTypeVar, sigma: float | tuple[float, float, float]
) -> TensorTypeVar:
    data_np = convert.to_np(data)
    result = scipy.ndimage.gaussian_filter(data_np, sigma=sigma)
    return convert.astype(result, data, cast=True)


@builder.register("GaussianBlur")
@typechecked
@attrs.mutable
class GaussianBlur(DataTransform):
    sigma: float | Distribution

    frozen: bool = False

    sigma_distr: Distribution = attrs.field(init=False)

    prepared_sigma: float | None = attrs.field(init=False, default=None)

    def __attrs_post_init__(self):
        self.sigma_distr = to_distribution(self.sigma)

    def prepare(self) -> None:
        self.prepared_sigma = self.sigma_distr()

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if self.frozen:
            assert self.prepared_sigma is not None
            sigma = self.prepared_sigma
        else:
            sigma = self.sigma_distr()
        return apply_gaussian_filter(data, sigma)
