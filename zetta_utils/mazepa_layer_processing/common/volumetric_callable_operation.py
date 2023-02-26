from __future__ import annotations

import copy
from typing import Callable, Generic, Sequence, TypeVar

import attrs
import torch
from typing_extensions import ParamSpec

from zetta_utils import builder, mazepa, tensor_ops
from zetta_utils.geometry import Vec3D
from zetta_utils.layer import IndexChunker
from zetta_utils.layer.volumetric import VolumetricIndex, VolumetricLayer

from . import ChunkedApplyFlowSchema
from .callable_operation import _process_callable_kwargs

P = ParamSpec("P")
IndexT = TypeVar("IndexT", bound=VolumetricIndex)


@builder.register("VolumetricCallableOperation")
@mazepa.taskable_operation_cls
@attrs.mutable
class VolumetricCallableOperation(Generic[P]):
    """
    Wrapper that converts a volumetric processing callable to a taskable operation .
    Adds support for pad/crop_pad and destination index resolution change.

    :param fn: Callable that will perform data processing
    """

    fn: Callable[P, torch.Tensor]
    crop_pad: Sequence[int] = (0, 0, 0)
    res_change_mult: Sequence[float] = (1, 1, 1)
    input_crop_pad: Sequence[int] = attrs.field(init=False)
    operation_name: str | None = None

    def get_operation_name(self):
        if hasattr(self.fn, "get_display_name"):
            result = self.fn.get_display_name()
        elif hasattr(self.fn, "__name__"):
            result = self.fn.__name__
        else:
            result = type(self.fn).__name__
            if result == "function":
                result = "Unspecified Function"
        return result

    def get_input_resolution(self, dst_resolution: Vec3D) -> Vec3D:
        return dst_resolution / Vec3D(*self.res_change_mult)

    def with_added_crop_pad(self, crop_pad: Vec3D[int]) -> VolumetricCallableOperation[P]:
        return attrs.evolve(self, crop_pad=Vec3D[int](*self.crop_pad) + crop_pad)

    def __attrs_post_init__(self):
        input_crop_pad_raw = Vec3D[int](*self.crop_pad) * Vec3D[float](*self.res_change_mult)
        for e in input_crop_pad_raw:
            if not isinstance(e, int) and not e.is_integer():
                raise ValueError(
                    f"Destination layer crop_pad of {self.crop_pad} with resolution change "
                    f"multiplier of {self.res_change_mult} results in non-integer "
                    f"final input crop of {input_crop_pad_raw}."
                )
        self.input_crop_pad = input_crop_pad_raw.int()

    def __call__(  # pylint: disable=keyword-arg-before-vararg
        self,
        idx: VolumetricIndex,
        dst: VolumetricLayer,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        assert len(args) == 0

        idx_input = copy.deepcopy(idx)
        idx_input.resolution = self.get_input_resolution(idx.resolution)
        idx_input_padded = idx_input.padded(Vec3D[int](*self.crop_pad))
        task_kwargs = _process_callable_kwargs(idx_input_padded, kwargs)
        result_raw = self.fn(**task_kwargs)
        # Data crop amount is determined by the index pad and the
        # difference between the resolutions of idx and dst_idx
        dst_data = tensor_ops.crop(result_raw, crop=self.crop_pad)
        dst[idx] = dst_data


# TODO: remove as soon as `interpolate_flow` is cut and ComputeField is configured
# to use subchunkable
def build_chunked_volumetric_callable_flow_schema(
    fn: Callable[P, torch.Tensor],
    chunker: IndexChunker[IndexT],
    crop_pad: Vec3D[int] = Vec3D[int](0, 0, 0),
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
