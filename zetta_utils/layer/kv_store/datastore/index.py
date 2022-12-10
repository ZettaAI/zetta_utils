# pylint: disable=missing-docstring
from __future__ import annotations

from typing import Iterable, Optional, Tuple

import attrs
from google.cloud.datastore import Key

from zetta_utils import builder

from ..index import ConvertibleKVIndex, KVIndex, RawKVIndex


@builder.register("DataStoreIndex")
@attrs.mutable
class DataStoreIndex(KVIndex):
    project: str
    namespace: str
    kind: str
    idx_raw: ConvertibleKVIndex
    attributes: Optional[Tuple[str]] = None

    @classmethod
    def default_convert(cls, idx_raw: RawKVIndex) -> DataStoreIndex:
        raise NotImplementedError()

    def _key(self, raw_key: str) -> Key:
        return Key(self.kind, raw_key, project=self.project, namespace=self.namespace)

    def keys(self) -> Iterable[Key]:

        # simple string index
        if isinstance(self.idx_raw, str):
            return [self._key(self.idx_raw)]

        # multi-valued index, without attributes
        if isinstance(self.idx_raw, list):
            return [self._key(i) for i in self.idx_raw]

        # index, with attributes
        if isinstance(self.idx_raw, tuple):
            self.idx_raw, self.attributes = self.idx_raw
        return self.keys()
