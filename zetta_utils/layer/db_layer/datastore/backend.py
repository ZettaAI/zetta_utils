"""gcloud datastore backend"""
from __future__ import annotations

from copy import deepcopy
from typing import List, Optional, Union

import attrs
from google.cloud.datastore import Client, Entity, Key
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.layer import Backend

from .. import DataT, DBIndex


def _get_data_from_entities(idx: DBIndex, entities: List[Entity]) -> DataT:
    data = []
    for i in range(idx.get_size()):
        col_keys = idx.col_keys[i]
        start = i * len(col_keys)
        end = start + len(col_keys)
        row_data = {}
        for j, ent in enumerate(entities[start:end]):
            row_data[col_keys[j]] = ent[col_keys[j]]
        data.append(row_data)
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

    def _get_keys_or_entities(
        self, idx: DBIndex, data: Optional[DataT] = None
    ) -> Union[List[Key], List[Entity]]:
        keys = []
        entities = []
        for i, row_key in enumerate(idx.row_keys):
            parent_key = self.client.key("Row", row_key)
            for col_key in idx.col_keys[i]:
                child_key = self.client.key("Column", col_key, parent=parent_key)
                if data is None:
                    keys.append(child_key)
                else:
                    entity = Entity(key=child_key, exclude_from_indexes=(col_key,))
                    try:
                        entity[col_key] = data[i][col_key]
                        entities.append(entity)
                    except KeyError:
                        ...
        return keys if data is None else entities

    @property
    def client(self) -> Client:
        if self._client is None:
            self._client = Client(project=self.project, namespace=self.namespace)
        return self._client

    def read(self, idx: DBIndex) -> DataT:
        keys = self._get_keys_or_entities(idx)
        entities = self.client.get_multi(keys)
        return _get_data_from_entities(idx, entities)

    def write(self, idx: DBIndex, data: DataT):
        entities = self._get_keys_or_entities(idx, data=data)
        self.client.put_multi(entities)

    def with_changes(self, **kwargs) -> DatastoreBackend:
        """Currently not typed. See `Layer.with_backend_changes()` for the reason."""
        implemented_keys = ["namespace", "project"]
        for k in kwargs:
            if k not in implemented_keys:
                raise KeyError(f"key {k} received, expected one of {implemented_keys}")
        return attrs.evolve(
            deepcopy(self), namespace=kwargs["namespace"], project=kwargs.get("project")
        )

    def get_name(self) -> str:  # pragma: no cover
        return self.client.base_url
