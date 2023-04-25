# pylint: disable=missing-docstring
import random
from typing import Callable, Literal, Sequence, TypeVar

import attrs
import torch
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.layer import JointIndexDataProcessor
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


@builder.register("ComposedAugment")
@typechecked
@attrs.frozen
class ComposedAugment(JointIndexDataProcessor):
    augments: Sequence[JointIndexDataProcessor]

    def process_index(
        self, idx: VolumetricIndex, mode: Literal["read", "write"]
    ) -> VolumetricIndex:
        result = idx
        for augment in reversed(self.augments):
            result = augment.process_index(result, mode)
        return result

    def process_data(
        self, data: dict[str, torch.Tensor], mode: Literal["read", "write"]
    ) -> dict[str, torch.Tensor]:
        result = data
        for augment in self.augments:
            result = augment.process_data(result, mode)
        return result
