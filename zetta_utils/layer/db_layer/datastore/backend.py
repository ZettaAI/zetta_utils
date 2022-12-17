"""gcloud datastore backend"""

from typing import List, Optional, Union

import attrs
from google.cloud.datastore import Client, Entity, Key
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.layer import Backend

from .. import DataT, DBIndex


def _get_keys_or_entities(
    idx: DBIndex, data: Optional[DataT] = None
) -> Union[List[Key], List[Entity]]:
    keys = []
    entities = []
    col_keys = next(idx.col_keys)
    for i, row_key in enumerate(idx.row_keys):
        parent_key = Key("Row", row_key)
        if len(col_keys) == 1 and col_keys[0] == "value":
            if data is None:
                keys.append(parent_key)
            else:
                entity = Entity(key=parent_key, exclude_from_indexes=("value",))
                entity["value"] = data[i]["value"]
                entities.append(entity)
        else:
            for j, col_key in enumerate(col_keys):
                child_key = Key("Column", col_key, parent=parent_key)
                if data is None:
                    keys.append(child_key)
                else:
                    entity = Entity(key=child_key, exclude_from_indexes=("value",))
                    entity["value"] = data[i * len(col_keys) + j][col_key]
                    entities.append(entity)
    return keys if data is None else entities


def _get_data_from_entities(idx: DBIndex, entities: List[Entity]) -> DataT:
    data = []
    col_keys = next(idx.col_keys)
    for i in range(idx.row_keys_count):
        if len(col_keys) == 1 and col_keys[0] == "value":
            try:
                data.append({"value": entities[i]["value"]})
            except KeyError:
                ...
        else:
            start = i * len(col_keys)
            end = start + len(col_keys)
            row_data = {}
            for j, ent in enumerate(entities[start:end]):
                try:
                    row_data[col_keys[j]] = ent["value"]
                except KeyError:
                    ...
    return data


@builder.register("DatastoreBackend")
@typechecked
@attrs.mutable
class DatastoreBackend(Backend[DBIndex, DataT]):
    """
    Backend for IO on a given google datastore `namespace`.
    `namespace` is similar to a database.
    `project` defaults to `gcloud config get-value project` if not specified.
    """

    namespace: str
    project: Optional[str] = None
    _client: Optional[Client] = None

    @property
    def client(self) -> Client:
        if self._client is None:
            self._client = Client(project=self.project, namespace=self.namespace)
        return self._client

    def read(self, idx: DBIndex) -> DataT:
        keys = _get_keys_or_entities(idx)
        entities = self.client.get_multi(keys)
        return _get_data_from_entities(idx, entities)

    def write(self, idx: DBIndex, data: DataT):
        entities = _get_keys_or_entities(idx, data=data)
        self.client.put_multi(entities)

    def get_name(self) -> str:  # pragma: no cover
        return self.client.base_url
