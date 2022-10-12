from typing import Any
from random import randint

import attrs

from typeguard import typechecked

from zetta_utils import builder
from .base import SampleIndexer


@builder.register("RandomIndexer")
@typechecked
@attrs.frozen
class RandomIndexer(SampleIndexer):  # pragma: no cover # No conditionals, under 3 LoC
    """Indexer which wraps a PieceIndexer to index at random.
    Does NOT guarantee any coverage.

    :param indexer: PieceIndexer to be used.

    """

    inner_indexer: SampleIndexer

    def __len__(self):
        num_samples = len(self.inner_indexer)
        return num_samples

    def __call__(self, idx: int) -> Any:
        """Yield a sample index from an indexer given a dummy index.

        :param idx: Integer sample index, kept for compatibility even
        though it is unused.
        :return: Index of the type used by the wrapped PieceIndexer.

        """
        rand_idx = randint(0, len(self.inner_indexer) - 1)
        return self.inner_indexer(rand_idx)
