# pylint: disable=missing-docstring
import random
from typing import Callable, TypeVar

from typeguard import typechecked

R = TypeVar("R")


@typechecked
def prob_aug(aug: Callable[..., R]) -> Callable[..., R]:
    def wrapper(*args, prob: float = 1.0, **kwargs) -> R:
        if len(args) > 0:
            raise RuntimeError(
                "Only keyword arguments are allowed for to probabilistic augmentation "
                "application. "
                f"Received: args {args}, kwargs {kwargs}"
            )
        if "data" not in kwargs:
            raise RuntimeError(
                "No `data` argument provided to probabilistic augmentation application. "
                f"Received: kwargs {kwargs}"
            )
        result = kwargs["data"]
        coin = random.uniform(0, 1)
        if coin < prob:
            result = aug(*args, **kwargs)

        return result

    return wrapper
