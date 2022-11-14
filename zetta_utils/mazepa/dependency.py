from typing import Collection

import attrs
from typeguard import typechecked


@typechecked
@attrs.frozen
class Dependency:
    ids: Collection[str] = attrs.field(factory=set)

    def is_barrier(self):
        return len(self.ids) == 0
