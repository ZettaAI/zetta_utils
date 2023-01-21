from os import path
from typing import Any, Callable, Generic, List, Literal, Optional, TypeVar, Union

import attrs
from torch import Tensor
from typing_extensions import ParamSpec

from zetta_utils import builder, log, mazepa
from zetta_utils.bbox import BBox3D
from zetta_utils.layer.volumetric import VolumetricIndex, VolumetricLayer
from zetta_utils.typing import IntVec3D, Vec3D

from ..operation_protocols import BlendableOpProtocol
from .blendable_apply_flow import BlendableApplyFlowSchema
from .volumetric_callable_operation import VolumetricCallableOperation

logger = log.get_logger("zetta_utils")

IndexT = TypeVar("IndexT")
P = ParamSpec("P")
R_co = TypeVar("R_co", covariant=True)
T = TypeVar("T")


@mazepa.taskable_operation_cls
@attrs.mutable
class DelegatedSubchunkedOperation(Generic[P, R_co]):
    """
    An operation that delegates to a FlowSchema.
    """

    flow_schema: BlendableApplyFlowSchema[P, Any]
    executor: mazepa.Executor = mazepa.Executor()

    def get_input_resolution(self, dst_resolution: Vec3D) -> Vec3D:  # pylint: disable=no-self-use
        return dst_resolution

    def __call__(
        self, idx: VolumetricIndex, dst: VolumetricLayer, *args: P.args, **kwargs: P.kwargs
    ) -> None:
        self.executor(self.flow_schema(idx, dst, *args, **kwargs))


def expand_if_singleton(arg: Union[T, List[T]], length: int) -> List[T]:
    if isinstance(arg, list):
        if len(arg) == length:
            return arg
        else:
            raise ValueError(f"cannot expand a list of length {len(arg)} to {length}")
    return [arg for _ in range(length)]


@builder.register(
    "build_subchunkable_apply_flow",
    cast_to_vec3d=["dst_resolution"],
    cast_to_intvec3d=[
        "fn_or_op_crop_pad",
        "blend_pads",
        "crop_pads",
        "processing_chunk_sizes",
        "max_reduction_chunk_sizes",
    ],
)
def build_subchunkable_apply_flow(  # pylint: disable=keyword-arg-before-vararg, line-too-long, too-many-locals, too-many-branches
    bbox: BBox3D,
    dst: VolumetricLayer,
    dst_resolution: Vec3D,
    temp_layers_dirs: Union[str, List[str]],
    processing_chunk_sizes: List[IntVec3D],
    crop_pads: Union[IntVec3D, List[IntVec3D]] = IntVec3D(0, 0, 0),
    blend_pads: Union[IntVec3D, List[IntVec3D]] = IntVec3D(0, 0, 0),
    blend_modes: Union[
        Literal["linear", "quadratic"], List[Literal["linear", "quadratic"]]
    ] = "linear",
    fn_or_op_crop_pad: IntVec3D = IntVec3D(0, 0, 0),
    fn: Optional[Callable[P, Tensor]] = None,
    op: Optional[BlendableOpProtocol[P, Tensor]] = None,
    max_reduction_chunk_sizes: Optional[Union[IntVec3D, List[IntVec3D]]] = None,
    *args: P.args,
    **kwargs: P.kwargs,
) -> mazepa.Flow:

    idx = VolumetricIndex(resolution=dst_resolution, bbox=bbox)

    if fn is not None and op is not None:
        raise ValueError("Cannot take both `fn` and `op`; please choose one or the other.")
    if fn is None and op is None:
        raise ValueError("Need exactly one of `fn` and `op`; received neither.")
    if fn is not None:
        level0_op = VolumetricCallableOperation(fn, crop_pad=fn_or_op_crop_pad)
    elif op is not None:
        # mypy seems to be confused by protocol
        level0_op = op.with_added_crop_pad(fn_or_op_crop_pad)  # type:ignore
    num_levels = len(processing_chunk_sizes)
    if max_reduction_chunk_sizes is None:
        max_reduction_chunk_sizes = processing_chunk_sizes

    for arg in [
        max_reduction_chunk_sizes,
        crop_pads,
        blend_pads,
        blend_modes,
        temp_layers_dirs,
    ]:
        if isinstance(arg, list):
            if not len(arg) == num_levels:
                raise ValueError(
                    "The arguments `max_reduction_chunk_sizes` (optional), `blend_pads`,"
                    " `crop_pads`, `blend_modes`, `temp_layers_dirs`"
                    " must be singletons or"
                    " lists where the lengths math the `processing_chunk_sizes`."
                )

    max_reduction_chunk_sizes = expand_if_singleton(max_reduction_chunk_sizes, num_levels)
    blend_pads = expand_if_singleton(blend_pads, num_levels)
    crop_pads = expand_if_singleton(crop_pads, num_levels)
    blend_modes = expand_if_singleton(blend_modes, num_levels)
    temp_layers_dirs = expand_if_singleton(temp_layers_dirs, num_levels)

    """
    Check that the sizes are correct. Note that the the order has to be reversed for indexing.
    At level l (where processing_chunk_sizes[num_levels] = idx.stop-idx.start):
    processing_chunk_sizes[l+1] + 2*crop_pads[l] + 2*blend_pads[l+1] % processing_chunk_sizes[l] == 0
    blend_pads[l] * 2 <= processing_chunk_sizes[l]
    """
    logger.info("Checking the arguments given.")
    for level in range(num_levels - 1, -1, -1):
        i = num_levels - level - 1
        if i == 0:
            processing_chunk_size_higher = idx.shape
            blend_pad_higher = IntVec3D(0, 0, 0)
        else:
            processing_chunk_size_higher = processing_chunk_sizes[i - 1]
            blend_pad_higher = blend_pads[i - 1]
        processing_chunk_size = processing_chunk_sizes[i]
        crop_pad = crop_pads[i]
        blend_pad = blend_pads[i]

        processing_region = processing_chunk_size_higher + 2 * crop_pad + 2 * blend_pad_higher

        if processing_region % processing_chunk_size != IntVec3D(0, 0, 0):
            n_region_chunks = processing_region / processing_chunk_size
            n_region_chunks_rounded = IntVec3D(*(round(e) for e in n_region_chunks))
            if processing_region % n_region_chunks_rounded == Vec3D(0, 0, 0):
                rec_processing_chunk_size = IntVec3D(
                    *(processing_region / n_region_chunks_rounded)
                )
                rec_str = f"Recommendation for `processing_chunk_size[level]`: {rec_processing_chunk_size}"
            else:
                rec_str = (
                    "Unable to recommend processing_chunk_size[level]: try using a more divisible"
                    " `crop_pad[level]` and/or `blend_pad[level+1]`."
                )
            error_str = (
                "At each level (where the 0-th level is the smallest), the"
                " `processing_chunk_size[level+1]` + 2*`crop_pad[level]` + 2*`blend_pad[level+1]` must be"
                f" evenly divisible by the `processing_chunk_size[level]`.\n\nAt level {level}, received:\n"
                f"`processing_chunk_size[level+1]`: {processing_chunk_size_higher}\n"
                f"`crop_pad[level]`: {crop_pad}\n"
                f"`blend_pad[level+1]`: {blend_pad_higher}\n"
                f"Size of the region to be processed for the level: {processing_region}\n"
                f"Which must be (but is not) divisible by: `processing_chunk_size[level]`: {processing_chunk_size}\n\n"
            )
            raise ValueError(error_str + rec_str)
        if not blend_pad * 2 <= processing_chunk_size:
            raise ValueError(
                "At each level (where the 0-th level is the smallest), the `blend_pad[level]` must"
                f" be at most half of the `processing_chunk_size[level]`. At level {level}, received:\n"
                f"`blend_pad[level]`: {blend_pad}\n"
                f"`processing_chunk_size[level]`: {processing_chunk_size}"
            )

    """
    Basic building blocks where the work gets done, at the very bottom
    """
    flow_schema = BlendableApplyFlowSchema(
        operation=level0_op,
        processing_chunk_size=processing_chunk_sizes[-1],
        max_reduction_chunk_size=max_reduction_chunk_sizes[-1],
        dst_resolution=dst_resolution,
        crop_pad=crop_pads[-1],
        blend_pad=blend_pads[-1],
        blend_mode=blend_modes[-1],
        temp_layers_dir=path.join(temp_layers_dirs[-1], "chunks_level_0"),
    )

    """
    Iteratively build the hierarchy of schemas
    """
    for level in range(1, num_levels):
        flow_schema = BlendableApplyFlowSchema(
            operation=DelegatedSubchunkedOperation(  # type:ignore #readability over typing here
                flow_schema,
            ),
            processing_chunk_size=processing_chunk_sizes[-level - 1],
            max_reduction_chunk_size=max_reduction_chunk_sizes[-level - 1],
            dst_resolution=dst_resolution,
            crop_pad=crop_pads[-level - 1],
            blend_pad=blend_pads[-level - 1],
            blend_mode=blend_modes[-level - 1],
            temp_layers_dir=path.join(temp_layers_dirs[-level - 1], f"chunks_level_{level}"),
        )

    return flow_schema(idx, dst, *args, **kwargs)
