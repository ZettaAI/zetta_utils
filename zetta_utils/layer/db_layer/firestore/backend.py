"""gcloud datastore backend"""

from __future__ import annotations

import sys
from copy import copy, deepcopy
from typing import Any, Optional

import attrs
import numpy as np
from google.api_core.exceptions import GoogleAPICallError
from google.cloud.firestore import And, Client, FieldFilter, Or
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.layer.db_layer import DBBackend, DBDataT, DBIndex, DBRowDataT

MAX_KEYS_PER_REQUEST = 1000
TENACITY_IGNORE_EXC = (KeyError, RuntimeError, TypeError, ValueError, GoogleAPICallError)


@builder.register("FirestoreBackend")
@typechecked
@attrs.mutable
class FirestoreBackend(DBBackend):
    """
    Backend for IO on a given `collection` in a google firestore `database` .

    `collection` is a group of related documents/rows.

    `database` the google firestore database id.

    `project` defaults to `gcloud config get-value project` if not specified.
    """

    collection: str
    database: Optional[str] = None
    project: Optional[str] = None
    _client: Optional[Client] = None

    @property
    def client(self) -> Client:
        if self._client is None:
            self._client = Client(project=self.project, database=self.database)
        return self._client

    @property
    def name(self) -> str:  # pragma: no cover
        return self.client._target  # pylint: disable=protected-access

    def __deepcopy__(self, memo) -> FirestoreBackend:
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

    def __contains__(self, idx: str) -> bool:
        print(idx)
        doc_ref = self.client.collection(self.collection).document(idx)
        print(doc_ref)
        return doc_ref.get().exists

    def __len__(self) -> int:  # pragma: no cover # no emulator support
        collection_ref = self.client.collection(self.collection)
        count_query = collection_ref.count()
        results = count_query.get()
        return int(results[0][0].value)

    @retry(
        retry=retry_if_not_exception_type(TENACITY_IGNORE_EXC),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def read(self, idx: DBIndex) -> DBDataT:
        if len(idx) == 1:
            if idx.row_keys[0] not in self:
                raise KeyError(idx.row_keys[0])
        refs = [self.client.collection(self.collection).document(k) for k in idx.row_keys]
        snapshots = self.client.get_all(refs)
        results = [snapshot.to_dict() if snapshot.exists else {} for snapshot in snapshots]
        for r in results:
            r.pop("_id_nonunique")
        return results

    @retry(
        retry=retry_if_not_exception_type(TENACITY_IGNORE_EXC),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def write(self, idx: DBIndex, data: DBDataT):
        doc_refs = [self.client.collection(self.collection).document(k) for k in idx.row_keys]
        bulk_writer = self.client.bulk_writer()
        for ref, row_data in zip(doc_refs, data):
            row_data = copy(row_data)
            row_data["_id_nonunique"] = np.random.randint(sys.maxsize)
            bulk_writer.set(ref, row_data, merge=True)
        bulk_writer.flush()

    @retry(
        retry=retry_if_not_exception_type(TENACITY_IGNORE_EXC),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def clear(self, idx: DBIndex | None = None):
        """
        Delete rows from the database.

        If index provided, delete rows from the index; else, delete all rows.
        """
        bulk_writer = self.client.bulk_writer()
        if idx:
            doc_refs = [self.client.collection(self.collection).document(k) for k in idx.row_keys]
        else:
            collection_ref = self.client.collection(self.collection)
            doc_refs = collection_ref.list_documents()
        for ref in doc_refs:
            bulk_writer.delete(ref)
        bulk_writer.flush()

    def keys(
        self,
        column_filter: dict[str, list] | None = None,  # pylint: disable=unused-argument
    ) -> list[str]:
        """
        Not implemented, firestore does not yet support `keys_only` queries.
        """
        raise NotImplementedError()

    def query(
        self,
        column_filter: dict[str, list] | None = None,
        return_columns: tuple[str, ...] = (),  # pylint: disable=unused-argument
    ) -> dict[str, DBRowDataT]:
        """
        Fetch list of rows that match given filters.

        `column_filter` is a dict of column names with list of values to filter.
            If None, all rows are returned.
        """
        collection_ref = self.client.collection(self.collection)
        if column_filter:
            _filters = []
            for _key, _values in column_filter.items():
                _filters.extend([FieldFilter(_key, "==", v) for v in _values])
            if len(_filters) == 1:
                _q = collection_ref.where(filter=_filters[0])
            else:  # pragma: no cover # no emulator support for composite queries
                _q = collection_ref.where(filter=Or(_filters))
            snapshots = list(_q.stream())
        else:
            refs = collection_ref.list_documents()
            snapshots = self.client.get_all(refs)
        result = {}
        for snapshot in snapshots:
            result[snapshot.id] = snapshot.to_dict()
            result[snapshot.id].pop("_id_nonunique")
        return result

    def get_batch(
        self,
        batch_number: int,
        avg_rows_per_batch: int,
        return_columns: tuple[str, ...] = (),  # pylint: disable=unused-argument
    ) -> dict[str, DBRowDataT]:
        """
        Fetch a batch of rows from the db layer. Rows are assigned a uniform random int.

        `batch_number` used to determine the starting offset of the batch to return.

        `avg_rows_per_batch` approximate number of rows returned per batch.
            Also used to determine the total number of batches.
        """
        if len(self) == 0:
            return {}

        if len(self) < avg_rows_per_batch:
            return self.query()  # return all rows

        if batch_number * avg_rows_per_batch >= len(self):
            start = batch_number * avg_rows_per_batch
            raise IndexError(f"{start} is out of bounds [0 {len(self)}).")

        scale = avg_rows_per_batch / len(self)
        scaled_batch_size = int(scale * sys.maxsize)
        offset_start = batch_number * scaled_batch_size
        offset_end = (batch_number + 1) * scaled_batch_size
        offset_end = sys.maxsize if offset_end > sys.maxsize else offset_end

        filters = [
            FieldFilter("_id_nonunique", ">=", offset_start),
            FieldFilter("_id_nonunique", "<", offset_end),
        ]
        collection_ref = self.client.collection(self.collection)
        _query = collection_ref.where(filter=And(filters))
        snapshots = list(_query.stream())
        result = {}
        for snapshot in snapshots:
            result[snapshot.id] = snapshot.to_dict()
            result[snapshot.id].pop("_id_nonunique")
        return result

    def with_changes(self, **kwargs) -> FirestoreBackend:
        """Currently not typed. See `Layer.with_backend_changes()` for the reason."""
        implemented_keys = ["collection", "project"]
        for k in kwargs:
            if k not in implemented_keys:
                raise KeyError(f"key {k} received, expected one of {implemented_keys}")
        return attrs.evolve(
            deepcopy(self),
            collection=kwargs["collection"],
            database=kwargs.get("database"),
            project=kwargs.get("project"),
        )
