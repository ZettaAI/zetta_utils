from typing import Optional, Protocol, TypeVar, runtime_checkable

from typing_extensions import ParamSpec

from zetta_utils import mazepa
from zetta_utils.layer.volumetric import VolumetricIndex, VolumetricLayer
from zetta_utils.typing import Vec3D

P = ParamSpec("P")
R_co = TypeVar("R_co", covariant=True)
IndexT_contra = TypeVar("IndexT_contra", contravariant=True)


@runtime_checkable
class MultiresOpProtocol(Protocol[P, R_co]):
    def get_input_resolution(self, dst_resolution: Vec3D) -> Vec3D:
        ...

    def __call__(
        self,
        idx: VolumetricIndex,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R_co:
        ...

    def make_task(
        self,
        *args,
        **kwargs,
        #idx: VolumetricIndex,
        #*args: P.args,
        #**kwargs: P.kwargs,
    ) -> mazepa.Task[R_co]:
        ...


@runtime_checkable
class ComputeFieldOpProtocol(Protocol[R_co]):
    def get_input_resolution(self, dst_resolution: Vec3D) -> Vec3D:
        ...

    def __call__(
        self,
        idx: VolumetricIndex,
        dst: VolumetricLayer,
        src: VolumetricLayer,
        tgt: VolumetricLayer,
        src_field: Optional[VolumetricLayer] = None,
        tgt_field: Optional[VolumetricLayer] = None,
    ) -> R_co:
        ...

    def make_task(
        self,
        idx: VolumetricIndex,
        dst: VolumetricLayer,
        src: VolumetricLayer,
        tgt: VolumetricLayer,
        src_field: Optional[VolumetricLayer] = None,
        tgt_field: Optional[VolumetricLayer] = None,
    ) -> mazepa.Task[R_co]:
        ...


class ChunkableOpProtocol(Protocol[P, IndexT_contra, R_co]):
    def __call__(
        self,
        idx: IndexT_contra,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R_co:
        ...

    def make_task(
        self,
        *args,
        **kwargs,
        #idx: ,
        #*args: P.args,
        #**kwargs: P.kwargs,
    ) -> mazepa.Task[R_co]:
        ...

class BlendableOpProtocol(Protocol[P, R_co]):
    def get_input_resolution(self, dst_resolution: Vec3D) -> Vec3D:
        ...

    def __call__(
        self,
        idx: VolumetricIndex,
        dst: VolumetricLayer,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R_co:
        ...

    def make_task(
        self,
        *args,
        **kwargs,
        #idx: VolumetricIndex,
        #dst: VolumetricLayer,
        #*args: P.args,
        #**kwargs: P.kwargs,
    ) -> mazepa.Task[R_co]:
        ...
