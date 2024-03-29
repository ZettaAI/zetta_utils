"""gcloud datastore backend"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any, Optional, Union

import attrs
from google.cloud.datastore import Client, Entity, Key, query
from tenacity import retry, stop_after_attempt, wait_exponential
from typeguard import typechecked

from zetta_utils import builder

from .. import DBBackend, DBDataT, DBIndex


def _get_data_from_entities(idx: DBIndex, entities: list[Entity]) -> DBDataT:
    row_entities = defaultdict(list)
    for entity in entities:
        row_entities[entity.key.parent.id_or_name].append(entity)

    data = []
    for row_key in idx.row_keys:
        row_data = {}
        for entity in row_entities[row_key]:
            row_data[entity.key.id_or_name] = entity[entity.key.id_or_name]
        data.append(row_data)
    return data


@builder.register("DatastoreBackend")
@typechecked
@attrs.mutable
class DatastoreBackend(DBBackend):
    """
    Backend for IO on a given google datastore `namespace`.

    `namespace` is similar to a database.

    `project` defaults to `gcloud config get-value project` if not specified.
    """

    namespace: str
    project: Optional[str] = None
    database: Optional[str] = None
    _client: Optional[Client] = None
    _exclude_from_indexes: tuple[str, ...] = ()

    def __deepcopy__(self, memo) -> DatastoreBackend:
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Skip _client, because it's unpickleable
        for field in attrs.fields_dict(cls):
            if field == "_client":
                setattr(result, field, None)
            else:
                value = getattr(self, field)
                setattr(result, field, deepcopy(value, memo))

        return result

    def __getstate__(self):
        state = attrs.asdict(self)
        state["_client"] = None
        return state

    def __setstate__(self, state: dict[str, Any]):
        for field in attrs.fields_dict(self.__class__):
            setattr(self, field, state[field])

    def _get_keys_or_entities(
        self, idx: DBIndex, data: Optional[DBDataT] = None
    ) -> Union[list[Key], list[Entity]]:
        keys = []
        entities = []
        for i, row_key in enumerate(idx.row_keys):
            parent_key = self.client.key("Row", row_key)
            for col_key in idx.col_keys[i]:
                child_key = self.client.key("Column", col_key, parent=parent_key)
                if data is None:
                    keys.append(child_key)
                else:
                    entity = Entity(key=child_key, exclude_from_indexes=self.exclude_from_indexes)
                    try:
                        entity[col_key] = data[i][col_key]
                        entities.append(entity)
                    except KeyError:
                        ...
        return keys if data is None else entities

    @property
    def client(self) -> Client:
        if self._client is None:
            self._client = Client(
                project=self.project, namespace=self.namespace, database=self.database
            )
        return self._client

    def query(self, column_filter: dict[str, list] | None = None) -> list[str]:
        """
        Fetch list of keys that match given filters.
        `column_filter` is a dict of column names with list of values to filter.
        """
        _query = self.client.query(kind="Column")
        _query.keys_only()
        if column_filter:
            for _key, _values in column_filter.items():
                _filters = [query.PropertyFilter(_key, "=", v) for v in _values]
                if len(_filters) == 1:
                    _query.add_filter(filter=_filters[0])
                else:  # pragma: no cover
                    _query.add_filter(filter=query.Or(_filters))
        entities = list(_query.fetch())
        return [entity.key.parent.id_or_name for entity in entities]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def read(self, idx: DBIndex) -> DBDataT:
        keys = self._get_keys_or_entities(idx)
        entities = self.client.get_multi(keys)
        return _get_data_from_entities(idx, entities)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def write(self, idx: DBIndex, data: DBDataT):
        entities = self._get_keys_or_entities(idx, data=data)
        self.client.put_multi(entities)

    def exists(self, idx: DBIndex) -> bool:
        keys = self._get_keys_or_entities(idx)
        for _key in keys:
            if self.client.get(_key):
                return True
        return False

    def with_changes(self, **kwargs) -> DatastoreBackend:
        """Currently not typed. See `Layer.with_backend_changes()` for the reason."""
        implemented_keys = ["namespace", "project"]
        for k in kwargs:
            if k not in implemented_keys:
                raise KeyError(f"key {k} received, expected one of {implemented_keys}")
        return attrs.evolve(
            deepcopy(self), namespace=kwargs["namespace"], project=kwargs.get("project")
        )

    @property
    def name(self) -> str:  # pragma: no cover
        return self.client.base_url

    @property
    def exclude_from_indexes(self) -> tuple[str, ...]:  # pragma: no cover
        return self._exclude_from_indexes

    @exclude_from_indexes.setter
    def exclude_from_indexes(self, exclude: tuple[str, ...]) -> None:  # pragma: no cover
        self._exclude_from_indexes = exclude
