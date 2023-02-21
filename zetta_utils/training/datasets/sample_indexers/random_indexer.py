import random
from typing import Any

import attrs
from typeguard import typechecked

from zetta_utils import builder

from .base import SampleIndexer


@builder.register("RandomIndexer")
@typechecked
@attrs.mutable
class RandomIndexer(SampleIndexer):  # pragma: no cover # No conditionals, under 3 LoC
    """Indexer randomizes the order at which `inner_indexer` samples are pulled.

    :param indexer: SampleIndexer to be randomized.

    """

    inner_indexer: SampleIndexer
    order: list[int] = attrs.field(init=False)

    def __attrs_post_init__(self):
        self.order = list(range(0, len(self.inner_indexer)))
        random.shuffle(self.order)

    def __len__(self):
        num_samples = len(self.inner_indexer)
        return num_samples

    def __call__(self, idx: int) -> Any:
        """Yield a sample index from an indexer given a dummy index.

        :param idx: Integer sample index, kept for compatibility even
        though it is unused.
        :return: Index of the type used by the wrapped PieceIndexer.

        """
        rand_idx = self.order[idx]
        return self.inner_indexer(rand_idx)
