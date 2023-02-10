import bisect
from itertools import accumulate
from typing import Any, Sequence

import attrs
from typeguard import typechecked

from zetta_utils import builder

from .base import SampleIndexer


@builder.register("ChainIndexer")
@typechecked
@attrs.frozen
class ChainIndexer(SampleIndexer):
    """
    Iterates over a sequence of inner indexers.

    :param inner_indexer: Sequence of inner indexers.
    """

    inner_indexer: Sequence[SampleIndexer]
    num_samples: list[int] = attrs.field(init=False)

    def __attrs_post_init__(self):
        # Use `__setattr__` to keep the object frozen.
        num_samples = [0] + list(accumulate(len(indexer) for indexer in self.inner_indexer))
        object.__setattr__(self, "num_samples", num_samples)

    def __len__(self):
        return self.num_samples[-1]

    def __call__(self, idx: int) -> Any:
        """Yield a sample index from an indexer given a index.

        :param idx: Integer sample index.
        :return: Index of the type used by the wrapped inner indexer.
        """
        if idx not in range(0, len(self)):
            raise ValueError(f"idx expected to be in range [0, {len(self)}), but got {idx}.")

        inner_indexer = bisect.bisect_right(self.num_samples, idx) - 1
        inner_index = idx - self.num_samples[inner_indexer]

        return self.inner_indexer[inner_indexer](inner_index)
