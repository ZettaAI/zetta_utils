from typing import Optional, Tuple
from random import randint

import attrs

from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.training.datasets.sample_indexers import SampleIndexer
from zetta_utils.typing import Vec3D


@builder.register("RandomIndexer")
@typechecked
@attrs.frozen
class RandomIndexer(SampleIndexer):  # pragma: no cover
    """Indexer which wraps a SampleIndexer to index at random.
    Does NOT guarantee any coverage.

    :param indexer: SampleIndexer to be used.

    """

    indexer: SampleIndexer

    def __len__(self):
        num_samples = len(self.indexer)
        return num_samples

    def __call__(self, idx: int) -> Tuple[Optional[Vec3D], slice, slice, slice]:
        """Translate a sample index to a volumetric region in space.

        :param idx: Integer sample index, kept for compatibility even
        though it is unused.
        :return: Volumetric index for the training sample patch, including
            ``self.desired_resolution`` and the slice representation of the region
            at ``self.index_resolution``.

        """
        rand_idx = randint(0, len(self.indexer) - 1)
        return self.indexer(rand_idx)
