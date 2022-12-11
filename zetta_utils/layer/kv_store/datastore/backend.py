"""gcloud datastore backend"""

from typing import List, Mapping, Optional, TypeVar, Union

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
    project: Optional[str] = None
    _client: Optional[Client] = None

    @property
    def client(self) -> Client:
        if self._client is None:
            self._client = Client(project=self.project)
        return self._client

    def read(self, idx: DataStoreIndex) -> Union[ValueT, MultiValueT]:
        assert (
            self.project == idx.project
        ), f"Client project {self.project} and data project {idx.project} do not match."
        ents = self.client.get_multi(idx.keys)

        if len(idx.keys) == 1:
            return dict(ents[0].items()) if idx.attributes is None else ents[0].get("value")

        if idx.attributes is None:
            return [e.get("value") for e in ents]

        _attrs = set(idx.attributes)
        return [{k: v for k, v in e.items() if k in _attrs} for e in ents]

    def write(self, idx: DataStoreIndex, value: Union[ValueT, MultiValueT]):
        assert (
            self.project == idx.project
        ), f"Client project {self.project} and data project {idx.project} do not match."

        if isinstance(value, list):
            values = []
            for v in value:
                assert isinstance(v, (str, Mapping))
                values.append(v if isinstance(v, Mapping) else {"value": v})
        else:
            assert len(idx.keys) == 1, f"Found {len(idx.keys)} keys but only one value."
            assert isinstance(v, (str, Mapping))
            values = [value if isinstance(value, Mapping) else {"value": value}]

        self._write_entities(idx.keys, values)

    def get_name(self) -> str:  # pragma: no cover
        return self.client.base_url

    def _write_entities(self, keys: List[Key], values: List[Mapping[str, str]]):
        entities: List[Entity] = []
        for k, v in zip(keys, values):
            entity = Entity(k)
            entity.update(v)
            entities.append(entity)
        self.client.put_multi(entities)
