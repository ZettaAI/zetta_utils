# pylint: disable=missing-docstring
from __future__ import annotations

from typing import List, Optional, Tuple

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

    _keys: Optional[List[Key]] = None

    @classmethod
    def default_convert(cls, idx_raw: RawKVIndex) -> DataStoreIndex:
        raise NotImplementedError()

    def _make_key(self, raw_key: str) -> Key:
        return Key(self.kind, raw_key, project=self.project, namespace=self.namespace)

    def keys(self) -> List[Key]:

        if self._keys is not None:
            return self._keys

        # simple string index
        if isinstance(self.idx_raw, str):
            self._keys = [self._make_key(self.idx_raw)]
            return self._keys

        # multi-valued index, without attributes
        if isinstance(self.idx_raw, list):
            self._keys = [self._make_key(i) for i in self.idx_raw]
            return self._keys

        # index, with attributes
        if isinstance(self.idx_raw, tuple):
            self.idx_raw, self.attributes = self.idx_raw
        return self.keys()
