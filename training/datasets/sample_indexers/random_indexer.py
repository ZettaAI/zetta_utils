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
    :param replacement: Samples are drawn on-demand with replacement if ``True``.
    """

    inner_indexer: SampleIndexer
    replacement: bool = False
    order: list[int] = attrs.field(init=False)

    def __attrs_post_init__(self):
        if self.replacement:
            self.order = []
        else:
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
        if self.replacement:
            rand_idx = random.randint(0, len(self) - 1)
        else:
            rand_idx = self.order[idx]
        return self.inner_indexer(rand_idx)
