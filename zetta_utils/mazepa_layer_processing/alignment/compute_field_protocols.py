from typing import Optional, Protocol, TypeVar, runtime_checkable

from typing_extensions import ParamSpec

from zetta_utils import mazepa
from zetta_utils.layer.volumetric import VolumetricIndex, VolumetricLayer
from zetta_utils.typing import Vec3D

P = ParamSpec("P")
R_co = TypeVar("R_co", covariant=True)


@runtime_checkable
class ComputeFieldOperation(Protocol[R_co]):
    def get_input_resolution(self, dst_resolution: Vec3D) -> Vec3D:
        ...

    def __call__(
        self,
        idx: VolumetricIndex,
        dst: VolumetricLayer,
        src: VolumetricLayer,
        tgt: VolumetricLayer,
        src_field: Optional[VolumetricLayer] = None,
    ) -> R_co:
        ...

    def make_task(
        self,
        idx: VolumetricIndex,
        dst: VolumetricLayer,
        src: VolumetricLayer,
        tgt: VolumetricLayer,
        src_field: Optional[VolumetricLayer] = None,
    ) -> mazepa.Task[R_co]:
        ...
