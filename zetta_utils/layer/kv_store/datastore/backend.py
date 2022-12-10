"""gcloud datastore backend"""

from typing import List, Mapping, TypeVar, Union

import attrs
from google.cloud.datastore import Client, Entity, Key
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.layer import LayerBackend

from .index import DataStoreIndex

DataT = TypeVar("DataT", bound=str)
DataAndAttributesT = Mapping[str, str]
ValueT = Union[DataT, DataAndAttributesT]
MultiValueT = Union[List[DataT], List[DataAndAttributesT]]


@builder.register("DatastoreBackend")
@typechecked
@attrs.mutable
class DatastoreBackend(LayerBackend[DataStoreIndex, Union[ValueT, MultiValueT]]):
    client: Client = Client()

    def read(self, idx: DataStoreIndex) -> Union[ValueT, MultiValueT]:
        ...

    def write(self, idx: DataStoreIndex, value: Union[ValueT, MultiValueT]):
        if isinstance(value, list):
            values = []
            for v in value:
                assert isinstance(v, (str, Mapping))
                values.append(v if isinstance(v, Mapping) else {"value": v})
            self._write_entities(idx.keys, values)
        else:
            assert len(idx.keys) == 1, f"Number of keys {len(idx.keys)} but value is single."
            assert isinstance(v, (str, Mapping))
            value = value if isinstance(value, Mapping) else {"value": value}
            self._write_entities(idx.keys, [value])

    def get_name(self) -> str:  # pragma: no cover
        return self.client.base_url

    def _write_entities(self, keys: List[Key], values: List[Mapping[str, str]]):
        entities: List[Entity] = []
        for k, v in zip(keys, values):
            entity = Entity(k)
            entity.update(v)
            entities.append(entity)
        self.client.put_multi(entities)
