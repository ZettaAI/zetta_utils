from typing import TypeVar, Any, Protocol
from typing_extensions import ParamSpec
import mazepa
from zetta_utils.layer import LayerIndex

IndexT_contra = TypeVar("IndexT_contra", bound=LayerIndex, contravariant=True)
T = TypeVar("T")
P = ParamSpec("P")
P1 = ParamSpec("P1")
R_co = TypeVar("R_co", covariant=True)


class LayerProcessorTaskMaker(Protocol[IndexT_contra, P, R_co]):
    def __call__(
        self,
        idx: IndexT_contra,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R_co:
        ...

    def make_task(  # pylint: disable=unused-argument,no-self-use
        self,
        idx: IndexT_contra,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> mazepa.Task[P, R_co]:
        ...


class LayerProcessorJob(Protocol[IndexT_contra, P]):
    def __init__(
        self,
        idx: IndexT_contra,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R_co:
        ...

    def make_task(  # pylint: disable=unused-argument,no-self-use
        self,
        idx: IndexT_contra,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> mazepa.Task[P, R_co]:
        ...
