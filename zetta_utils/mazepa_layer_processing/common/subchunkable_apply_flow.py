from copy import deepcopy
from os import path
from typing import Callable, Generic, List, Literal, Optional, TypeVar, Union

import attrs
from torch import Tensor
from typing_extensions import ParamSpec

from zetta_utils import builder, log, mazepa
from zetta_utils.geometry import BBox3D, IntVec3D, Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex, VolumetricLayer

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
class DelegatedSubchunkedOperation(Generic[P]):
    """
    An operation that delegates to a FlowSchema.
    """

    flow_schema: BlendableApplyFlowSchema[P, None]

    def get_input_resolution(self, dst_resolution: Vec3D) -> Vec3D:  # pylint: disable=no-self-use
        return dst_resolution

    def __call__(
        self, idx: VolumetricIndex, dst: VolumetricLayer, *args: P.args, **kwargs: P.kwargs
    ) -> None:
        mazepa.Executor(do_dryrun_estimation=False, show_progress=False)(
            self.flow_schema(idx, dst, *args, **kwargs)
        )


def expand_if_singleton(arg: Union[T, List[T]], length: int) -> List[T]:
    if isinstance(arg, list):
        if len(arg) == length:
            return arg
        else:
            raise ValueError(f"cannot expand a list of length {len(arg)} to {length}")
    return [arg for _ in range(length)]


@builder.register("build_subchunkable_apply_flow")
def build_subchunkable_apply_flow(  # pylint: disable=keyword-arg-before-vararg, line-too-long, too-many-locals, too-many-branches, too-many-statements
    dst: VolumetricLayer,
    dst_resolution: Vec3D,
    temp_layers_dirs: Union[str, List[str]],
    processing_chunk_sizes: List[IntVec3D],
    processing_crop_pads: Union[IntVec3D, List[IntVec3D]] = IntVec3D(0, 0, 0),
    processing_blend_pads: Union[IntVec3D, List[IntVec3D]] = IntVec3D(0, 0, 0),
    processing_blend_modes: Union[
        Literal["linear", "quadratic"], List[Literal["linear", "quadratic"]]
    ] = "linear",
    fov_crop_pad: IntVec3D = IntVec3D(0, 0, 0),
    start_coord: Optional[IntVec3D] = None,
    end_coord: Optional[IntVec3D] = None,
    coord_resolution: Optional[Vec3D] = None,
    bbox: Optional[BBox3D] = None,
    fn: Optional[Callable[P, Tensor]] = None,
    op: Optional[BlendableOpProtocol[P, None]] = None,
    max_reduction_chunk_sizes: Optional[Union[IntVec3D, List[IntVec3D]]] = None,
    allow_cache_up_to_level: int = 0,
    *args: P.args,
    **kwargs: P.kwargs,
) -> mazepa.Flow:
    if bbox is None:
        if start_coord is None or end_coord is None or coord_resolution is None:
            raise ValueError(
                "`bbox` was not supplied, so `start_coord`, end_coord`, and"
                " `coord_resolution` has to be specified."
            )
        bbox = BBox3D.from_coords(
            start_coord=start_coord, end_coord=end_coord, resolution=coord_resolution
        )
    else:
        if start_coord is not None or end_coord is not None or coord_resolution is not None:
            raise ValueError(
                "`bbox` was supplied, so `start_coord`, end_coord`, and"
                " `coord_resolution` cannot be specified."
            )
    idx = VolumetricIndex(resolution=dst_resolution, bbox=bbox)

    num_levels = len(processing_chunk_sizes)
    if max_reduction_chunk_sizes is None:
        max_reduction_chunk_sizes = processing_chunk_sizes

    for arg in [
        max_reduction_chunk_sizes,
        processing_crop_pads,
        processing_blend_pads,
        processing_blend_modes,
        temp_layers_dirs,
    ]:
        if isinstance(arg, list):
            if not len(arg) == num_levels:
                raise ValueError(
                    "The arguments `max_reduction_chunk_sizes` (optional), `processing_blend_pads`,"
                    " `processing_crop_pads`, `processing_blend_modes`, `temp_layers_dirs`"
                    " must be singletons or"
                    " lists where the lengths math the `processing_chunk_sizes`."
                )

    max_reduction_chunk_sizes = expand_if_singleton(max_reduction_chunk_sizes, num_levels)
    processing_blend_pads = expand_if_singleton(processing_blend_pads, num_levels)
    processing_crop_pads = expand_if_singleton(processing_crop_pads, num_levels)
    processing_blend_modes = expand_if_singleton(processing_blend_modes, num_levels)
    temp_layers_dirs = expand_if_singleton(temp_layers_dirs, num_levels)

    if fn is not None and op is not None:
        raise ValueError("Cannot take both `fn` and `op`; please choose one or the other.")
    if fn is None and op is None:
        raise ValueError("Need exactly one of `fn` and `op`; received neither.")
    # mypy seems to be confused by protocol
    if fn is not None:
        level0_op = VolumetricCallableOperation(fn, crop_pad=processing_crop_pads[-1])
    elif op is not None:
        level0_op = op.with_added_crop_pad(processing_crop_pads[-1])  # type:ignore

    """
    Check that the sizes are correct. Note that the the order has to be reversed for indexing.
    At level l (where processing_chunk_sizes[num_levels] = idx.stop-idx.start):
    processing_chunk_sizes[l+1] + 2*processing_crop_pads[l+1] + 2*processing_blend_pads[l+1] % processing_chunk_sizes[l] == 0
    processing_blend_pads[l] * 2 <= processing_chunk_sizes[l]
    """
    logger.info("Checking the arguments given.")
    for level in range(num_levels - 1, -1, -1):
        i = num_levels - level - 1
        if i == 0:
            processing_chunk_size_higher = idx.shape
            processing_blend_pad_higher = IntVec3D(0, 0, 0)
            processing_crop_pad_higher = fov_crop_pad
        else:
            processing_chunk_size_higher = processing_chunk_sizes[i - 1]
            processing_blend_pad_higher = processing_blend_pads[i - 1]
            processing_crop_pad_higher = processing_crop_pads[i - 1]
        processing_chunk_size = processing_chunk_sizes[i]
        processing_blend_pad = processing_blend_pads[i]

        processing_region = (
            processing_chunk_size_higher
            + 2 * processing_crop_pad_higher
            + 2 * processing_blend_pad_higher
        )

        if processing_region % processing_chunk_size != IntVec3D(0, 0, 0):
            n_region_chunks = processing_region / processing_chunk_size
            n_region_chunks_rounded = IntVec3D(*(round(e) for e in n_region_chunks))
            if processing_region % n_region_chunks_rounded == Vec3D(0, 0, 0):
                rec_processing_chunk_size = (processing_region / n_region_chunks_rounded).int()
                rec_str = f"Recommendation for `processing_chunk_size[level]`: {rec_processing_chunk_size}"
            else:
                rec_str = (
                    "Unable to recommend processing_chunk_size[level]: try using a more divisible"
                    " `processing_crop_pad[level+1]` and/or `processing_blend_pad[level+1]`."
                )
            error_str = (
                "At each level (where the 0-th level is the smallest), the"
                " `processing_chunk_size[level+1]` + 2*`processing_crop_pad[level+1]` + 2*`processing_blend_pad[level+1]` must be"
                f" evenly divisible by the `processing_chunk_size[level]`.\n\nAt level {level}, received:\n"
                f"`processing_chunk_size[level+1]`:\t\t\t\t{processing_chunk_size_higher}\n"
                f"`processing_crop_pad[level+1]` (`fov_crop_pad` for the top level):\t{processing_crop_pad_higher}\n"
                f"`processing_blend_pad[level+1]`:\t\t\t\t{processing_blend_pad_higher}\n"
                f"Size of the region to be processed for the level: {processing_region}\n"
                f"Which must be (but is not) divisible by: `processing_chunk_size[level]`: {processing_chunk_size}\n\n"
            )
            raise ValueError(error_str + rec_str)
        if not processing_blend_pad * 2 <= processing_chunk_size:
            raise ValueError(
                "At each level (where the 0-th level is the smallest), the `processing_blend_pad[level]` must"
                f" be at most half of the `processing_chunk_size[level]`. At level {level}, received:\n"
                f"`processing_blend_pad[level]`: {processing_blend_pad}\n"
                f"`processing_chunk_size[level]`: {processing_chunk_size}"
            )

    """
    Append fov_crop pads into one list
    """
    fov_crop_pads = [fov_crop_pad] + processing_crop_pads[:-1]
    """
    Check for no checkerboarding
    """
    for i in range(0, num_levels - 1):
        if fov_crop_pads[i] == IntVec3D(0, 0, 0) and processing_blend_pads[i] == IntVec3D(0, 0, 0):
            logger.info(
                f"Level {num_levels - i - 1}: Chunks will not be checkerboarded since `processing_crop_pad[level+1]`"
                f" (`fov_crop_pad` for the top level) and `processing_blend_pad[level]` are both {IntVec3D(0, 0, 0)}."
            )
    if processing_blend_pads[num_levels - 1] == IntVec3D(0, 0, 0):
        logger.info(
            f"Level 0: Chunks will not be checkerboarded since `processing_blend_pad[level]` is {IntVec3D(0, 0, 0)}."
        )
    if fov_crop_pads[0] == IntVec3D(0, 0, 0) and processing_blend_pads[0] == IntVec3D(0, 0, 0):
        logger.info(
            "Since checkerboarding is skipped at the top level, the FOV is required to be chunk-aligned."
        )
        dst = attrs.evolve(
            deepcopy(dst), backend=dst.backend.with_changes(enforce_chunk_aligned_writes=True)
        )
    else:
        dst = attrs.evolve(
            deepcopy(dst), backend=dst.backend.with_changes(enforce_chunk_aligned_writes=False)
        )

    """
    Basic building blocks where the work gets done, at the very bottom
    """
    flow_schema: BlendableApplyFlowSchema = BlendableApplyFlowSchema(
        op=level0_op,
        processing_chunk_size=processing_chunk_sizes[-1],
        max_reduction_chunk_size=max_reduction_chunk_sizes[-1],
        dst_resolution=dst_resolution,
        fov_crop_pad=fov_crop_pads[-1],
        processing_blend_pad=processing_blend_pads[-1],
        processing_blend_mode=processing_blend_modes[-1],
        temp_layers_dir=path.join(temp_layers_dirs[-1], "chunks_level_0"),
        allow_cache=(allow_cache_up_to_level >= 1),
        clear_cache_on_return=(allow_cache_up_to_level == 1),
    )

    """
    Iteratively build the hierarchy of schemas
    """
    for level in range(1, num_levels):
        flow_schema = BlendableApplyFlowSchema(
            op=DelegatedSubchunkedOperation(  # type:ignore #readability over typing here
                flow_schema,
            ),
            processing_chunk_size=processing_chunk_sizes[-level - 1],
            max_reduction_chunk_size=max_reduction_chunk_sizes[-level - 1],
            dst_resolution=dst_resolution,
            fov_crop_pad=fov_crop_pads[-level - 1],
            processing_blend_pad=processing_blend_pads[-level - 1],
            processing_blend_mode=processing_blend_modes[-level - 1],
            temp_layers_dir=path.join(temp_layers_dirs[-level - 1], f"chunks_level_{level}"),
            allow_cache=(allow_cache_up_to_level >= level + 1),
            clear_cache_on_return=(allow_cache_up_to_level == level + 1),
        )

    return flow_schema(idx, dst, *args, **kwargs)
