from typing import Optional, Iterable

import attrs
from typeguard import typechecked


@typechecked
@attrs.frozen
class Dependency:
    ids: Iterable[str] = attrs.field(factory=set)

    def is_barrier(self):
        return len(self.ids) == 0
