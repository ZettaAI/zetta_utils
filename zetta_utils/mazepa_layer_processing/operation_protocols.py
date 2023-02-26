from __future__ import annotations

from typing import Optional, Protocol, TypeVar, runtime_checkable

from typing_extensions import ParamSpec

from zetta_utils import mazepa
from zetta_utils.geometry import Vec3D
from zetta_utils.layer.volumetric import (
    VolumetricBasedLayerProtocol,
    VolumetricIndex,
    VolumetricLayer,
)

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
        idx: VolumetricIndex,
        *args: P.args,
        **kwargs: P.kwargs,
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
        idx: IndexT_contra,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> mazepa.Task[R_co]:
        ...


DstLayerT_contra = TypeVar(
    "DstLayerT_contra", bound=VolumetricBasedLayerProtocol, contravariant=True
)


@runtime_checkable
class VolumetricOpProtocol(Protocol[P, R_co, DstLayerT_contra]):
    def get_input_resolution(self, dst_resolution: Vec3D) -> Vec3D:
        ...

    def with_added_crop_pad(
        self, crop_pad: Vec3D[int]
    ) -> VolumetricOpProtocol[P, R_co, DstLayerT_contra]:
        ...

    def __call__(
        self,
        idx: VolumetricIndex,
        dst: DstLayerT_contra,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R_co:
        ...

    def make_task(
        self,
        idx: VolumetricIndex,
        dst: DstLayerT_contra,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> mazepa.Task[R_co]:
        ...
