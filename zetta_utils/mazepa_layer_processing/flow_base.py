from typing import TypeVar, Any, Protocol
from typing_extensions import ParamSpec
import mazepa
from zetta_utils.layer import LayerIndex

IndexT_contra = TypeVar("IndexT_contra", bound=LayerIndex, contravariant=True)
T = TypeVar("T")
P = ParamSpec("P")
R_co = TypeVar("R_co", covariant=True)


class LayerFlow(Protocol[P, R_co, ]):
    def __call__(
        self,
        idx: LayerIndex,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> :
        ...
