"""gcloud datastore backend"""

from typing import Any, TypeVar

import attrs
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.layer import LayerBackend, LayerIndex

IndexT = TypeVar("IndexT", bound=LayerIndex)


@builder.register("DatastoreBackend")
@typechecked
@attrs.mutable
class DatastoreBackend(LayerBackend[IndexT, Any]):
    ...
