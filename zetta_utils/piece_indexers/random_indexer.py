from typing import Any
from random import randint

import attrs

from typeguard import typechecked

from zetta_utils import builder
from .base import PieceIndexer


@builder.register("RandomIndexer")
@typechecked
@attrs.frozen
class RandomIndexer(PieceIndexer):  # pragma: no cover
    """Indexer which wraps a PieceIndexer to index at random.
    Does NOT guarantee any coverage.

    :param indexer: PieceIndexer to be used.

    """

    indexer: PieceIndexer

    def __len__(self):
        num_samples = len(self.indexer)
        return num_samples

    def __call__(self, idx: int) -> Any:
        """Yield a sample index from an indexer given a dummy index.

        :param idx: Integer sample index, kept for compatibility even
        though it is unused.
        :return: Index of the type used by the wrapped PieceIndexer.

        """
        rand_idx = randint(0, len(self.indexer) - 1)
        return self.indexer(rand_idx)
