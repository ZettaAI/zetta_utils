from typing import Generic, TypeVar, Dict, Any
from abc import ABC
import mazepa
from zetta_utils.layer import LayerIndex, Layer

IndexT = TypeVar("IndexT", bound=LayerIndex)
T = TypeVar("T")

# Needed for potential future spec error checking
class LayerProcessor(ABC, Generic[IndexT]):
    def __call__(self, layers: Dict[str, Layer[Any, IndexT]], idx: IndexT):
        ...

    def make_task(  # pylint: disable=unused-argument,no-self-use
        self, layers: Dict[str, Layer[Any, IndexT]], idx: IndexT
    ) -> mazepa.Task:
        ...
