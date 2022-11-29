from typing import Protocol, TypeVar

from typing_extensions import ParamSpec

from zetta_utils import mazepa
from zetta_utils.layer import LayerIndex

P = ParamSpec("P")
R_co = TypeVar("R_co", covariant=True)
IndexT_contra = TypeVar("IndexT_contra", bound=LayerIndex, contravariant=True)


class ChunkableTaskFactory(Protocol[P, IndexT_contra, R_co]):
    def make_task(
        self,
        idx: IndexT_contra,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> mazepa.Task[R_co]:
        ...

    def __call__(
        self,
        idx: IndexT_contra,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R_co:
        ...
