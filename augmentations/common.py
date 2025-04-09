# pylint: disable=missing-docstring
import random
from typing import Callable, TypeVar

from typeguard import typechecked

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
