from typing import Generic, TypeVar, Dict, Any
from zetta_utils.layer import LayerIndex, Layer

IndexT = TypeVar("IndexT", bound=LayerIndex)
# Needed for potential future spec error checking
class LayerProcessor(Generic[IndexT]):
    def __call__(self, layers: Dict[str, Layer[Any, IndexT]], idx: IndexT):
        pass
