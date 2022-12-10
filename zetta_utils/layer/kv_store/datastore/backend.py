"""gcloud datastore backend"""

from typing import List, Mapping, TypeVar, Union

import attrs
from google.cloud.datastore import Client, Entity
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
        # scalar key value
        if not isinstance(value, list):
            assert len(idx.keys) == 1, f"Number of keys {len(idx.keys)} but value is single."
            key = idx.keys[0]
            entity = Entity(key)
            value = value if isinstance(value, Mapping) else {"value": value}
            entity.update(value)
            self.client.put(entity)

    def get_name(self) -> str:  # pragma: no cover
        return self.client.base_url
