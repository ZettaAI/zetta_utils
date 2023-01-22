from __future__ import annotations

import copy
from typing import Callable, Generic, Optional, TypeVar

import attrs
import torch
from typing_extensions import ParamSpec

from zetta_utils import builder, mazepa, tensor_ops
from zetta_utils.geometry import IntVec3D, Vec3D
from zetta_utils.layer import IndexChunker, Layer
from zetta_utils.layer.volumetric import VolumetricIndex, VolumetricLayer

from . import ChunkedApplyFlowSchema

P = ParamSpec("P")
IndexT = TypeVar("IndexT", bound=VolumetricIndex)


@builder.register(
    "VolumetricCallableOperation", cast_to_vec3d=["res_change_mult"], cast_to_intvec3d=["crop_pad"]
)
@mazepa.taskable_operation_cls
@attrs.mutable
class VolumetricCallableOperation(Generic[P]):
    """
    Wrapper that converts a volumetric processing callable to a taskable operation .
    Adds support for pad/crop_pad and destination index resolution change.

    :param fn: Callable that will perform data processing
    """

    fn: Callable[P, torch.Tensor]
    crop_pad: IntVec3D = IntVec3D(0, 0, 0)
    res_change_mult: Vec3D = Vec3D(1, 1, 1)
    input_idx_pad: IntVec3D = attrs.field(init=False)

    def get_input_resolution(self, dst_resolution: Vec3D) -> Vec3D:
        return dst_resolution / self.res_change_mult

    def with_added_crop_pad(self, crop_pad: IntVec3D) -> VolumetricCallableOperation[P]:
        return attrs.evolve(self, crop_pad=self.crop_pad + crop_pad)

    def __attrs_post_init__(self):
        input_idx_pad_raw = self.crop_pad * self.res_change_mult
        for e in input_idx_pad_raw:
            if not e.is_integer():
                raise ValueError(
                    f"Destination layer crop_pad of {self.crop_pad} with resolution change "
                    f"multiplier of {self.res_change_mult} results in non-integer "
                    f"input index crop_pad of {input_idx_pad_raw}."
                )
        self.input_idx_pad = IntVec3D(*(int(e) for e in input_idx_pad_raw))

    def __call__(  # pylint: disable=keyword-arg-before-vararg
        self,
        idx: VolumetricIndex,
        dst: VolumetricLayer,
        crop_pad: Optional[IntVec3D] = None,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        assert len(args) == 0
        if crop_pad is not None:
            self.__attrs_post_init__()
        idx_input = copy.deepcopy(idx)
        idx_input.resolution = self.get_input_resolution(idx.resolution)
        idx_input_padded = idx_input.padded(self.input_idx_pad)

        task_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, Layer):
                task_kwargs[k] = v[idx_input_padded]
            else:
                task_kwargs[k] = v

        result_raw = self.fn(**task_kwargs)

        # Data crop amount is determined by the index pad and the
        # difference between the resolutions of idx and dst_idx
        dst_data = tensor_ops.crop(result_raw, crop=self.crop_pad)
        dst[idx] = dst_data


@builder.register(
    "build_chunked_volumetric_callable_flow_schema",
    cast_to_vec3d=["res_change_mult"],
    cast_to_intvec3d=["crop_pad"],
)
def build_chunked_volumetric_callable_flow_schema(
    fn: Callable[P, torch.Tensor],
    chunker: IndexChunker[IndexT],
    crop_pad: IntVec3D = IntVec3D(0, 0, 0),
    res_change_mult: Vec3D = Vec3D(1, 1, 1),
) -> ChunkedApplyFlowSchema[P, IndexT, None]:
    operation = VolumetricCallableOperation[P](
        fn=fn,
        crop_pad=crop_pad,
        res_change_mult=res_change_mult,
    )
    return ChunkedApplyFlowSchema[P, IndexT, None](
        chunker=chunker,
        operation=operation,  # type: ignore
    )
