from __future__ import annotations

import copy
from typing import Callable, Generic, Optional, TypeVar

import attrs
import torch
from typing_extensions import ParamSpec

from zetta_utils import builder, mazepa, tensor_ops
from zetta_utils.layer import IndexChunker, Layer
from zetta_utils.layer.volumetric import VolumetricIndex, VolumetricLayer
from zetta_utils.typing import IntVec3D, Vec3D

from . import ChunkedApplyFlowSchema

P = ParamSpec("P")
IndexT = TypeVar("IndexT", bound=VolumetricIndex)


@builder.register(
    "VolumetricCallableOperation", cast_to_vec3d=["res_change_mult"], cast_to_intvec3d=["crop"]
)
@mazepa.taskable_operation_cls
@attrs.mutable
class VolumetricCallableOperation(Generic[P]):
    """
    Wrapper that converts a volumetric processing callable to a taskable operation .
    Adds support for pad/crop and destination index resolution change.

    :param fn: Callable that will perform data processing
    """

    fn: Callable[P, torch.Tensor]
    crop: IntVec3D = IntVec3D(0, 0, 0)
    res_change_mult: Vec3D = Vec3D(1, 1, 1)
    input_idx_pad: IntVec3D = attrs.field(init=False)
    operation_base_name: Optional[str] = None

    def get_operation_name(  # pylint: disable=unused-argument
        self, idx: VolumetricIndex, dst: VolumetricLayer, *args: P.args, **kwargs: P.kwargs
    ) -> str:
        base = self.operation_base_name

        if base is None:
            if hasattr(base, "__name__"):
                base = self.fn.__name__
            else:
                base = type(self.fn).__name__
                if base == "function":
                    base = "Unspecified Function"

        return base

    def get_input_resolution(self, dst_resolution: Vec3D) -> Vec3D:
        return dst_resolution / self.res_change_mult

    def __attrs_post_init__(self):
        input_idx_pad_raw = self.crop * self.res_change_mult
        for e in input_idx_pad_raw:
            if not e.is_integer():
                raise ValueError(
                    f"Destination layer crop of {self.crop} with resolution change "
                    f"multiplier of {self.res_change_mult} results in non-integer "
                    f"input index crop of {input_idx_pad_raw}."
                )
        self.input_idx_pad = IntVec3D(*(int(e) for e in input_idx_pad_raw))

    def __call__(
        self, idx: VolumetricIndex, dst: VolumetricLayer, *args: P.args, **kwargs: P.kwargs
    ) -> None:
        assert len(args) == 0
        idx_input = copy.deepcopy(idx)
        idx_input.resolution = self.get_input_resolution(idx.resolution)
        idx_input_padded = idx_input.pad(self.input_idx_pad)

        task_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, Layer):
                task_kwargs[k] = v[idx_input_padded]
            else:
                task_kwargs[k] = v

        result_raw = self.fn(**task_kwargs)

        # Data crop ammount is determined by the index pad and the
        # difference between the resolutions of idx and dst_idx
        dst_data = tensor_ops.crop(result_raw, crop=self.crop)
        dst[idx] = dst_data


@builder.register(
    "build_chunked_volumetric_callable_flow_schema",
    cast_to_vec3d=["res_change_mult"],
    cast_to_intvec3d=["crop"],
)
def build_chunked_volumetric_callable_flow_schema(
    fn: Callable[P, torch.Tensor],
    chunker: IndexChunker[IndexT],
    crop: IntVec3D = IntVec3D(0, 0, 0),
    res_change_mult: Vec3D = Vec3D(1, 1, 1),
    operation_base_name: Optional[str] = None,
) -> ChunkedApplyFlowSchema[P, IndexT, None]:
    operation = VolumetricCallableOperation[P](
        fn=fn, crop=crop, res_change_mult=res_change_mult, operation_base_name=operation_base_name
    )
    return ChunkedApplyFlowSchema[P, IndexT, None](
        chunker=chunker,
        operation=operation,  # type: ignore
    )
