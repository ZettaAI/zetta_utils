from typing import Any

import attrs
from typeguard import typechecked

from zetta_utils import builder

from .base import SampleIndexer


@builder.register("LoopIndexer")
@typechecked
@attrs.frozen
class LoopIndexer(SampleIndexer):  # pragma: no cover # No conditionals, under 3 LoC
    """
    Loops over a inner indexer to match the desired number of samples.

    :param inner_indexer: Inner Indexer.
    :param desired_num_samples: Number of samples.

    """

    inner_indexer: SampleIndexer
    desired_num_samples: int

    def __len__(self):
        return self.desired_num_samples

    def __call__(self, idx: int) -> Any:
        """Yield a sample index from an indexer given a index.

        :param idx: Integer sample index.
        :return: Index of the type used by the wrapped inner indexer.
        """
        loop_idx = idx % len(self.inner_indexer)
        return self.inner_indexer(loop_idx)
