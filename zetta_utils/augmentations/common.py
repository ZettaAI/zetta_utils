# pylint: disable=missing-docstring
from __future__ import annotations

import random
from abc import abstractmethod
from typing import Any, Callable, Literal, Sequence, TypeVar

import attrs
import torch
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.layer import DataProcessor, JointIndexDataProcessor
from zetta_utils.layer.volumetric import VolumetricIndex

R = TypeVar("R")


@typechecked
def prob_aug(aug: Callable[..., R]) -> Callable[..., R]:
    def wrapper(*args, prob: float = 1.0, **kwargs) -> R:
        try:
            if len(args) == 0:
                result = kwargs["data"]
            else:
                result = args[0]
        except (KeyError, IndexError) as e:
            raise RuntimeError(
                "Input data to probabilistic augmentation application must be either "
                "provided as first positional argument, or as 'data' keyword without "
                "any positional arguments. "
                f"Received: args {args}, kwargs {kwargs}"
            ) from e
        coin = random.uniform(0, 1)
        if coin < prob:
            result = aug(*args, **kwargs)

        return result

    return wrapper


@typechecked
@attrs.mutable
class DataAugment(DataProcessor):
    prob: float

    def __call__(self, data: Any) -> Any:
        if random.uniform(0, 1) < self.prob:
            data = self.augment(data)
        return data

    @abstractmethod
    def augment(self, data: Any) -> Any:
        ...


@typechecked
@attrs.mutable
class JointIndexDataAugment(JointIndexDataProcessor):
    prob: float

    prepared_coin: bool | None = attrs.field(init=False, default=None)

    def process_index(
        self, idx: VolumetricIndex, mode: Literal["read", "write"]
    ) -> VolumetricIndex:
        self.prepared_coin = random.uniform(0, 1) < self.prob
        if self.prepared_coin:
            idx = self.augment_index(idx)
        return idx

    def process_data(
        self, data: dict[str, torch.Tensor], mode: Literal["read", "write"]
    ) -> dict[str, torch.Tensor]:
        assert self.prepared_coin is not None
        if self.prepared_coin:
            data = self.augment_data(data)
        self.prepared_coin = None
        return data

    @abstractmethod
    def augment_index(self, idx: VolumetricIndex) -> VolumetricIndex:
        ...

    @abstractmethod
    def augment_data(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        ...


@builder.register("ComposedAugment")
@typechecked
@attrs.frozen
class ComposedAugment(JointIndexDataProcessor):
    augments: Sequence[DataAugment | JointIndexDataAugment]

    def process_index(
        self, idx: VolumetricIndex, mode: Literal["read", "write"]
    ) -> VolumetricIndex:
        idx_proced = idx
        for augment in reversed(self.augments):
            if isinstance(augment, JointIndexDataAugment):
                idx_proced = augment.process_index(idx=idx_proced, mode=mode)
        return idx_proced

    def process_data(
        self, data: dict[str, torch.Tensor], mode: Literal["read", "write"]
    ) -> dict[str, torch.Tensor]:
        data_proced = data
        for augment in self.augments:
            if isinstance(augment, JointIndexDataAugment):
                data_proced = augment.process_data(data=data_proced, mode=mode)
            else:
                data_proced = augment(data_proced)
        return data_proced
