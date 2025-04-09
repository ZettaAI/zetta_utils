# pylint: disable=too-many-lines
from __future__ import annotations

import math
from collections.abc import Sequence as AbcSequence
from copy import deepcopy
from os import path
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

import attrs
from torch import Tensor
from typing_extensions import ParamSpec

from zetta_utils import builder, log, mazepa
from zetta_utils.common.pprint import lrpad, utcnow_ISO8601
from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.geometry.vec import VEC3D_PRECISION
from zetta_utils.layer.volumetric import VolumetricBasedLayerProtocol, VolumetricIndex
from zetta_utils.layer.volumetric.cloudvol.build import build_cv_layer
from zetta_utils.mazepa import SemaphoreType, id_generation
from zetta_utils.ng.link_builder import make_ng_link
from zetta_utils.typing import ensure_seq_of_seq

from ..operation_protocols import VolumetricOpProtocol
from .volumetric_apply_flow import VolumetricApplyFlowSchema
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

    flow_schema: VolumetricApplyFlowSchema[P, None]
    operation_name: str
    level: int

    def get_input_resolution(  # pylint: disable=no-self-use
        self, dst_resolution: Vec3D
    ) -> Vec3D:  # pragma: no cover
        return dst_resolution

    def get_operation_name(self) -> str:
        return f"Level {self.level} {self.operation_name}"

    def __call__(
        self,
        idx: VolumetricIndex,
        dst: VolumetricBasedLayerProtocol,
        *op_args: P.args,
        **op_kwargs: P.kwargs,
    ) -> None:
        mazepa.Executor(
            do_dryrun_estimation=False,
            show_progress=False,
        )(self.flow_schema(idx, dst, op_args, op_kwargs))


@builder.register("build_postpad_subchunkable_apply_flow")
def build_postpad_subchunkable_apply_flow(  # pylint: disable=keyword-arg-before-vararg, too-many-locals, too-many-branches, too-many-statements
    dst: VolumetricBasedLayerProtocol,
    dst_resolution: Sequence[float],
    processing_input_sizes: Sequence[Sequence[int]],
    processing_crop: Sequence[int] = (0, 0, 0),
    processing_blend: Sequence[int] = (0, 0, 0),
    processing_blend_mode: Literal["linear", "quadratic", "max", "defer"] = "quadratic",
    level_intermediaries_dirs: Sequence[str | None] | None = None,
    max_reduction_chunk_size: Sequence[int] | None = None,
    expand_bbox_processing: bool = True,
    expand_bbox_resolution: bool = False,
    allow_cache_up_to_level: int | None = None,
    print_summary: bool = True,
    generate_ng_link: bool = False,
    op_worker_type: str | None = None,
    reduction_worker_type: str | None = None,
    fn: Callable[P, Tensor] | None = None,
    fn_semaphores: Sequence[SemaphoreType] | None = None,
    op: VolumetricOpProtocol[P, None, Any] | None = None,
    op_args: Iterable = (),
    op_kwargs: Mapping[str, Any] | None = None,
    bbox: BBox3D | None = None,
    start_coord: Sequence[int] | None = None,
    end_coord: Sequence[int] | None = None,
    coord_resolution: Sequence | None = None,
    auto_bbox: bool = False,
    dst_tighten_bounds: bool = False,
) -> mazepa.Flow:
    """
    Wrapper for SubchunkableApplyFlow that treats the chunk sizes as post padding for
    blend and crop. The bottom level chunk is preserved, but all upper level chunk sizes are
    treated as upper bounds. Crop is only applied at the bottom level, and blend at all
    levels are identical.

    See main SubchunkableApplyFlow documentation for behaviour of most other arguments; the
    differences are highlighted here. Note that with this wrapper, the following arguments are
    always set, and ``dst`` is not optional:
    - auto_divisibility: True
    - processing_gap: (0,0,0)
    - expand_bbox_processing: True
    - expand_bbox_backend: False
    - shrink_processing_chunk: False
    - skip_intermediaries: False

    :param processing_input_sizes: The input size after padding for blending and cropping,
        at each subchunking level in X, Y, Z, from the largest to the smallest. Must be even.
        All upper levels are treated as upper bounds.
    :param processing_crop: Pixels to crop from the processing input at the bottom level.
    :param processing_blend: Pixels to crop and blend per processing chunk at each subchunking
        level, from each side. If the input size is ``1024, 1024, 1`` with no crop and
        ``4, 4, 0`` blend, then the unpadded outputs will be ``1016, 1016, 1`` for each chunk,
        and the 4 pixels at the edges will be blended with the 4 pixels from the neighbouring
        chunk. Must be smaller than or equal to quarter of the ``processing_chunk_input`` minus
        2 x crop in each dimension.
    :param processing_blend_mode: Which blend mode to use at each subchunking level. ``linear``
        sums the blended areas weighted linearly by the position. ``quadratic`` sums the
        blended areas weighted quadratically by the position.  ``max`` takes the maximum value
        of any layer in the overlap area.  ``defer`` can only be supplied as the
        blend mode for the top level, and skips the final reduction stage, leaving the
        final intermediary files for the user to handle. If ``defer`` is used, then
        ``skip_intermediaries`` cannot be used.

    """
    processing_crop_vec3d = Vec3D(*processing_crop)
    processing_blend_vec3d = Vec3D(*processing_blend)

    processing_chunk_sizes = [
        Vec3D(*size) - 2 * processing_blend_vec3d for size in processing_input_sizes
    ]
    processing_chunk_sizes[-1] -= 2 * processing_crop_vec3d
    processing_crop_pads = [Vec3D(0, 0, 0) for _ in processing_input_sizes]
    processing_crop_pads[-1] = processing_crop_vec3d
    processing_blend_pads = [processing_blend_vec3d for _ in processing_input_sizes]

    logger.info(
        "Converted from given post-pad input sizes and padding to pre-pad core chunk sizes:\n"
        "Received:\n\t`processing_input_sizes`:"
        f"\t{', '.join(str(tuple(size)) for size in processing_input_sizes)}\n"
        f"\t`processing_crop`:\t\t{tuple(processing_crop)}\t(at the bottom level)\n"
        f"\t`processing_blend`:\t\t{tuple(processing_blend)}\t(at all levels)\n"
        "As core chunk sizes, before padding for crop and blend:\n"
        f"\t`processing_chunk_sizes`:"
        "\t{', '.join(size.pformat() for size in processing_chunk_sizes)}\n"
        "The bottom level chunk size will be respected to maintain the input size of "
        f"{tuple(processing_input_sizes[-1])} while the other levels will be "
        "treated as upper bounds, fitting in as many chunks as possible."
    )

    return build_subchunkable_apply_flow(
        dst=dst,
        dst_resolution=dst_resolution,
        processing_chunk_sizes=processing_chunk_sizes,
        processing_gap=(0, 0, 0),
        processing_crop_pads=processing_crop_pads,
        processing_blend_pads=processing_blend_pads,
        processing_blend_modes=processing_blend_mode,
        level_intermediaries_dirs=level_intermediaries_dirs,
        skip_intermediaries=False,
        max_reduction_chunk_size=max_reduction_chunk_size,
        expand_bbox_resolution=expand_bbox_resolution,
        expand_bbox_backend=False,
        expand_bbox_processing=expand_bbox_processing,
        shrink_processing_chunk=False,
        auto_divisibility=True,
        allow_cache_up_to_level=allow_cache_up_to_level,
        op_worker_type=op_worker_type,
        reduction_worker_type=reduction_worker_type,
        print_summary=print_summary,
        generate_ng_link=generate_ng_link,
        fn=fn,
        fn_semaphores=fn_semaphores,
        op=op,
        op_args=op_args,
        op_kwargs=op_kwargs,
        bbox=bbox,
        start_coord=start_coord,
        end_coord=end_coord,
        coord_resolution=coord_resolution,
        auto_bbox=auto_bbox,
        dst_tighten_bounds=dst_tighten_bounds,
    )


@builder.register("build_subchunkable_apply_flow")
def build_subchunkable_apply_flow(  # pylint: disable=keyword-arg-before-vararg, too-many-locals, too-many-branches, too-many-statements
    dst: VolumetricBasedLayerProtocol | None,
    dst_resolution: Sequence[float],
    processing_chunk_sizes: Sequence[Sequence[int]],
    processing_gap: Sequence[int] = (0, 0, 0),
    processing_crop_pads: Sequence[int] | Sequence[Sequence[int]] = (0, 0, 0),
    processing_blend_pads: Sequence[int] | Sequence[Sequence[int]] = (0, 0, 0),
    processing_blend_modes: Union[
        Literal["linear", "quadratic", "max", "defer"],
        Sequence[Literal["linear", "quadratic", "max", "defer"]],
    ] = "quadratic",
    level_intermediaries_dirs: Sequence[str | None] | None = None,
    skip_intermediaries: bool = False,
    max_reduction_chunk_size: Sequence[int] | None = None,
    expand_bbox_resolution: bool = False,
    expand_bbox_backend: bool = False,
    expand_bbox_processing: bool = True,
    shrink_processing_chunk: bool = False,
    auto_divisibility: bool = False,
    allow_cache_up_to_level: int | None = None,
    op_worker_type: str | None = None,
    reduction_worker_type: str | None = None,
    print_summary: bool = True,
    generate_ng_link: bool = False,
    fn: Callable[P, Tensor] | None = None,
    fn_semaphores: Sequence[SemaphoreType] | None = None,
    op: VolumetricOpProtocol[P, None, Any] | None = None,
    op_args: Iterable = (),
    op_kwargs: Mapping[str, Any] | None = None,
    bbox: BBox3D | None = None,
    start_coord: Sequence[int] | None = None,
    end_coord: Sequence[int] | None = None,
    coord_resolution: Sequence | None = None,
    auto_bbox: bool = False,
    dst_tighten_bounds: bool = False,
) -> mazepa.Flow:
    """
    The helper constructor for a flow that applies any function or operation with a `Tensor`
    valued output in a chunkwise fashion, allowing for arbitrary subchunking with cropping and
    blending. Chunks are processed, written to an intermediary location temporarily, and then
    combined (``reduced``) with weights if necessary to produce the output.

    :param dst: The destination VolumetricBasedLayerProtocol.  May be None, in which case
        `skip_intermediaries` must be True, `expand_bbox_backend` must be False, and
        "defer" may not be used in `processing_blend_modes`.
    :param dst_resolution: The resolution of the destination VolumetricBasedLayerProtocol
        (or resolution to use for computation, even if `dst` is None).
    :param dst_tighten_bounds: Tighten the VolumetricBasedLayerProtocol's bounds to the
        given ``bbox`` before any expansions (whether given as a ``bbox`` or  ``start_coord``,
        ``end_coord``, and ``coord_resolution``) EXCEPT for ``expand_bbox_resolution``.
        If this is the case, then the processing will happen in the expanded bbox according to
        all expansion, though the writing will only happen within the tight bounds.
        Note that this only happens for the ``dst_resolution``, and may result in odd behaviour
        if the bounds at multiple resolutions are used in a subsequent processes. Requires ``dst``.
    :param processing_chunk_sizes: The base chunk size at each subchunking level in X, Y, Z,
        from the largest to the smallest. Subject to divisibility requirements (see bottom). When
        ``auto_divisibility`` is used, the chunk sizes other than the bottom level chunk size will
        be treated as an upper bound, and rounded down to satisfy divisibility. Must be even.
    :param processing_gap: Extra unprocessed space to be skipped between chunks at the top level.
        When used, blend_pad cannot be used at the top level. Cannot be used with
        ``auto_divisibility``, ``expand_bbox_backend``, ``expand_bbox_resolution``, or
        ``shrink_processing_chunk``.
    :param processing_crop_pads: Pixels to crop per processing chunk at each subchunking
        level in X, Y, Z, from the largest to the smallest. Affects divisibility requirements
        (see bottom). Given as a padding: ``(10, 10, 0)`` ``crop_pad`` with a ``(1024, 1024, 1)``
        ``processing_chunk_size`` means that ``(1044, 1044, 1)`` area will be processed and then
        cropped by ``(10, 10, 0)`` to output a ``(1024, 1024, 1)`` chunk.
    :param processing_blend_pads: Pixels to blend per processing chunk at each subchunking
        level in X, Y, Z, from the largest to the smallest. Affects divisibility requirements
        (see bottom).Given as a padding: ``(10, 10, 0)`` ``blend_pad`` with a ``(1024, 1024, 1)``
        ``processing_chunk_size`` means that ``(1044, 1044, 1)`` area will be processed and then
        be overlapped by ``(20, 20, 0)`` between each ``(1024, 1024, 1)`` chunk. Must be less
        than or equal to half of the ``processing_chunk_size`` in each dimension.
    :param processing_blend_modes: Which blend mode to use at each subchunking level. ``linear``
        sums the blended areas weighted linearly by the position. ``quadratic`` sums the
        blended areas weighted quadratically by the position.  ``max`` takes the maximum value
        of any layer in the overlap area.  ``defer`` can only be supplied as the
        blend mode for the top level, and skips the final reduction stage, leaving the
        final intermediary files for the user to handle. If ``defer`` is used, then
        ``skip_intermediaries`` cannot be used.
    :param max_reduction_chunk_size: The upper bounds of the size for chunks to be used for the
        reduction step. During the reduction step, backend chunks in the area to be reduced will
        be reduced in larger chunks that have been combined up to this limit. Reduction chunks
        are only used to combine already computed outputs, so larger is better to cut down on
        the number of tasks. Must be larger than both the `processing_chunk_size` for the top
        level, as well as the `dst` chunk size.
    :param level_intermediaries_dirs: Intermediary directories for temporary layers at each
        subchunking level, used for handling blending, cropping, and rechunking for backends.
        Only used if the level is using blending and/or if the level above has crop, or if it is
        the top level and ``skip_intermediaries`` has not been set. If running remotely, the top
        level ``must`` be also remote, as the worker performing the reduction step may be
        different from the worker that wrote the processing. The other levels are recommended to
        be local.
    :param skip_intermediaries: Skips all intermediaries. This means that no blending is allowed
        anywhere, and that only the bottom level may have crop. You MUST ensure that your output
        is aligned to the backend chunk yourself when this option is used. Cannot be used if
        ``defer`` is used in ``processing_blend_modes``.
    :param expand_bbox_resolution: Expands ``bbox`` (whether given as a ``bbox`` or
        ``start_coord``, ``end_coord``, and ``coord_resolution``) to be integral in the
        ``dst_resolution``. Cannot be used with ``processing_gap``.
    :param expand_bbox_backend: Expands ``bbox`` (whether given as a ``bbox`` or ``start_coord``,
        ``end_coord``, and ``coord_resolution``) to be aligned to the ``dst`` layer's backend
        chunk size and offset at ``dst_resolution``.  Requires ``bbox`` to be integral in
        ``dst_resolution``. Cannot be used with ``processing_gap``, ``expand_bbox_processing``, or
        ``auto_divisibility``.
    :param expand_bbox_processing: Expands ``bbox`` (whether given as a ``bbox`` or
        ``start_coord``, ``end_coord``, and ``coord_resolution``) to be an integer multiple of
        the top level ``processing_chunk_size``, holding the top left corner fixed.  Requires
        ``bbox`` to be integral in ``dst_resolution``. Cannot be used with
        ``expand_bbox_backend`` or ``shrink_processing_chunk``.
    :param shrink_processing_chunk: Shrinks the top level ``processing_chunk_size`` to fit the
        ``bbox``. Does not affect other levels, so divisibility requirements may be affected.
        Requires ``bbox`` to be integral in ``dst_resolution``. Cannot be used with
        ``processing_gap``, ``expand_bbox_processing``, or ``auto_divisibility``.
    :param auto_divisibility: Automatically chooses ``processing_chunk_sizes`` that are divisible,
        while respecting the bottom level ``processing_chunk_size`` as well as every level's
        ``processing_crop_pads`` and ``processing_blend_pads``. The user-provided
        ``processing_chunk_sizes`` are treated as an upper bound.  Requires ``bbox`` to be
        integral in ``dst_resolution``. Requires ``expand_bbox_processing``. Cannot be used with
        ``processing_gap``, ``expand_bbox_backend``, ``shrink_processing_chunk``.
    :param allow_cache_up_to_level: The subchunking level (smallest is 0) where the cache for
        different remote layers should be cleared after the processing is done. Recommended to
        keep this at the level of the largest subchunks (default).
    :param op_worker_type: The worker type required by the op. The subchunked tasks
        will be routed to only the workers that have the requested worker type.
    :param reduction_worker_type: The worker type required by the reduction process. The
        subchunked tasks will be routed to only the workers that have the requested worker type.
    :param print_summary: Whether a summary should be printed.
    :param generate_ng_link: Whether a neuroglancer link should be generated in the summary.
        Requires ``print_summary``.
    :param fn: The function to be run on each chunk. Cannot be used with ``op``.
    :param fn_semaphores: The semaphores to be acquired while the function runs. Supports
        ``read``, ``write``, ``cuda``, and ``cpu``. Cannot be used with ``op``.
    :param op: The operation to be run on each chunk. Cannot be used with ``fn``.
    :param op_args: Only used for typing. Do not use: will raise an exception if nonempty.
    :param op_kwargs: Any kwarguments taken by the ``fn`` or the ``op``.
        ``end_coord``, and ``coord_resolution``.
    :param bbox: The bounding box for the operation. Cannot be used with ``start_coord``,
        ``end_coord``, and ``coord_resolution``, or ``auto_bbox``.
    :param start_coord: The start coordinate for the bounding box. Must be used with
        ``end_coord`` and ``coord_resolution``; cannot be used with ``bbox`` or ``auto_bbox.
    :param end_coord: The end coordinate for the bounding box. Must be used with ``start_coord``
        and ``coord_resolution``; cannot be used with ``bbox`` or ``auto_bbox.
    :param coord_resolution: The resolution in which the coordinates are given for the bounding
        box. Must be used with ``start_coord`` and ``end_coord``; cannot be used with ``bbox``.
    :param auto_bbox: Sets the ``bbox`` to the bounds of ``dst`` at ``dst_resolution``. Note
        that this may result in a significant overhead if the ``dst`` bounds have not been
        tightened. Requires ``dst``. Cannot be used with ``bbox`` or ``start_coord``,
        ``end_coord``, and ``coord_resolution``.

    """

    dst_resolution_ = Vec3D(*dst_resolution)

    if dst is None:
        if not skip_intermediaries:
            raise ValueError("`skip_intermediaries` must be True when `dst` is None.")
        if expand_bbox_backend:
            raise ValueError("Cannot use `expand_bbox_backend` when `dst` is None.")
        if max_reduction_chunk_size is not None:
            raise ValueError("`max_reduction_chunk_size` is unused when `dst` is None.")
        if dst_tighten_bounds:
            raise ValueError("`dst_tighten_bounds` cannot be used when `dst` is None.")
        if auto_bbox:
            raise ValueError("`auto_bbox` cannot be used when `dst` is None.")

    bbox_ = parse_bbox(
        dst=dst,
        bbox=bbox,
        start_coord=start_coord,
        end_coord=end_coord,
        coord_resolution=coord_resolution,
        dst_resolution=dst_resolution_,
        auto_bbox=auto_bbox,
    )

    if fn is not None and op is not None:
        raise ValueError("Cannot take both `fn` and `op`; please choose one or the other.")
    if fn_semaphores is not None and op is not None:
        raise ValueError(
            "Cannot take both `fn_semaphores` and `op`; please use either `fn` with "
            "`fn_semaphores, or use `op`."
        )
    if fn is None and op is None:
        raise ValueError("Need exactly one of `fn` and `op`; received neither.")

    if op is not None:
        assert fn is None
        op_ = op
    else:
        assert fn is not None
        op_ = VolumetricCallableOperation[P](fn, fn_semaphores=fn_semaphores)

    op_kwargs_ = op_kwargs if op_kwargs is not None else {}

    if generate_ng_link and not print_summary:
        raise ValueError("Cannot use `generate_ng_link` when `print_summary=False`.")

    num_levels = len(processing_chunk_sizes)

    # Explicit checking here for prettier user eror
    seq_of_seq_arguments = {
        "processing_crop_pads": processing_crop_pads,
        "processing_blend_pads": processing_blend_pads,
        "processing_blend_modes": processing_blend_modes,
    }
    for k, v in seq_of_seq_arguments.items():
        if (
            not isinstance(v, str)
            and isinstance(v, AbcSequence)
            and len(v) > 0
            and isinstance(v[0], AbcSequence)
        ):
            if not len(v) == num_levels:
                raise ValueError(
                    f"If provided as a nested sequence, length of `{k}` must be equal to "
                    f"the number of subchunking levels, which is {num_levels} (inferred from "
                    f"`processing_chunk_sizes`). Got `len({k}) == {len(v)}`"
                )

    processing_blend_pads_ = ensure_seq_of_seq(processing_blend_pads, num_levels)
    processing_crop_pads_ = ensure_seq_of_seq(processing_crop_pads, num_levels)
    processing_blend_modes_ = ensure_seq_of_seq(processing_blend_modes, num_levels)

    if processing_blend_modes_[0] == "defer":
        if dst is None:
            raise ValueError(
                '`dst` cannot be None when `processing_blend_modes` is using "defer".'
            )
        if skip_intermediaries:
            raise ValueError('`skip_intermediaries` cannot be used with "defer".')
        logger.warning(
            'Because `processing_blend_modes` is "defer", `dst` will NOT '
            "contain the final output of subchunkable."
        )
        if dst and dst.write_procs != ():
            logger.warning(
                "Unblended intermediaries will already have the write_procs applied "
                "to them; another Flow, Operation, or Task that writes to `dst`, such "
                "as a naive copy, may apply the write_procs a second time."
            )
    for i in range(1, num_levels):
        if processing_blend_modes_[i] == "defer":
            raise ValueError(
                'Blending mode "defer" is only supported for the top level; '
                "please specify other blending modes for other levels."
            )

    if max_reduction_chunk_size is None:
        max_reduction_chunk_size_ = processing_chunk_sizes[0]
    else:
        if not Vec3D(*max_reduction_chunk_size) >= Vec3D(  # pylint: disable=unneeded-not
            *processing_chunk_sizes[0]
        ):
            raise ValueError(
                "`max_reduction_chunk_size` must be larger than or equal to the top level "
                f"`processing_chunk_size`; received {max_reduction_chunk_size}, which is "
                f"smaller than {processing_chunk_sizes[0]}."
            )
        if dst and not Vec3D(
            *max_reduction_chunk_size
        ) >= dst.backend.get_chunk_size(  # pylint: disable=unneeded-not
            dst_resolution_
        ):
            raise ValueError(
                "`max_reduction_chunk_size` must be larger than or equal to the `dst` backend's"
                f"`chunk_size` at `dst_resolution`; received {max_reduction_chunk_size}, which "
                "is smaller than the chunk size "
                f"{dst.backend.get_chunk_size(dst_resolution_)} at resolution "
                f"{dst_resolution}."
            )

        max_reduction_chunk_size_ = max_reduction_chunk_size

    if skip_intermediaries:
        if reduction_worker_type is not None:
            raise ValueError(
                "`reduction_worker_type` cannot be used when `skip_intermediaries` is True."
            )
        if level_intermediaries_dirs is not None:
            raise ValueError(
                "`level_intermediaries_dirs` was supplied even though "
                "`skip_intermediaries` = True."
            )
        if any(any(v != 0 for v in pad) for pad in processing_blend_pads_):
            raise ValueError(
                f"`processing_blend_pads` must be {[0, 0, 0]} in all levels when "
                "`skip_intermediaries` = True."
            )
        if any(any(v != 0 for v in pad) for pad in processing_crop_pads_[:-1]):
            raise ValueError(
                f"`processing_crop_pads` must be {[0, 0, 0]} in all levels except the "
                "bottom (smallest) when `skip_intermediaries` = True."
            )
    else:
        if level_intermediaries_dirs is not None and len(level_intermediaries_dirs) != num_levels:
            raise ValueError(
                f"`len(level_intermediaries_dirs)` != {num_levels}, where {num_levels} is the "
                "number of subchunking levels inferred from `processing_chunk_sizes`"
            )
        if level_intermediaries_dirs is None:
            raise ValueError(
                "`level_intermediaries_dirs` is required unless `skip_intermediaries` is used."
            )

    if level_intermediaries_dirs is not None:
        level_intermediaries_dirs_ = level_intermediaries_dirs
    else:
        level_intermediaries_dirs_ = [None for _ in range(num_levels)]

    if auto_divisibility is True and shrink_processing_chunk is True:
        raise ValueError(
            "`auto_divisibility` cannot be used with `shrink_processing_chunk`; "
            "Please choose at most one.",
        )
    if auto_divisibility is True and expand_bbox_processing is False:
        raise ValueError("`auto_divisibility` requires `expand_bbox_processing` to be `True`.")
    if auto_divisibility is True and expand_bbox_backend is True:
        raise ValueError(
            "`auto_divisibility` requires `expand_bbox_backend` to be `False`. "
            "This is because divisibility should take precedence over backend alignment."
        )
    if shrink_processing_chunk and expand_bbox_processing:
        raise ValueError(
            "`shrink_processing_chunk` and `expand_bbox_processing` cannot both be `True`; "
            "Please choose at most one.",
        )

    if allow_cache_up_to_level is None:
        allow_cache_up_to_level_ = num_levels - 1
    else:
        allow_cache_up_to_level_ = allow_cache_up_to_level

    if processing_gap is not None and Vec3D(*processing_gap) != Vec3D[int](0, 0, 0):
        if any(v % 2 != 0 for v in processing_gap):
            raise ValueError("`processing_gap` must be divisible by 2 in all dimensions.")
        if any(v != 0 for v in processing_blend_pads_[0]):
            raise ValueError(
                "`blend_pads` at top level must be zero when `processing_gap` is used."
            )
        if auto_divisibility:
            raise ValueError("`auto_divisibility` cannot be used with nonzero `processing_gap`.")
        if shrink_processing_chunk:
            raise ValueError(
                "`shrink_processing_chunk` cannot be used with nonzero `processing_gap`."
            )
        if expand_bbox_backend:
            raise ValueError("`expand_bbox_backend` cannot be used with nonzero `processing_gap`.")
        if expand_bbox_resolution:
            raise ValueError(
                "`expand_bbox_resolution` cannot be used with nonzero `processing_gap`."
            )

    assert len(list(op_args)) == 0
    return _build_subchunkable_apply_flow(
        dst=dst,
        dst_resolution=dst_resolution_,
        level_intermediaries_dirs=level_intermediaries_dirs_,
        dst_tighten_bounds=dst_tighten_bounds,
        skip_intermediaries=skip_intermediaries,
        processing_chunk_sizes=[Vec3D(*v) for v in processing_chunk_sizes],
        processing_gap=Vec3D(*processing_gap),
        processing_crop_pads=[Vec3D(*v) for v in processing_crop_pads_],
        processing_blend_pads=[Vec3D(*v) for v in processing_blend_pads_],
        processing_blend_modes=processing_blend_modes_,  # type: ignore # Literal gets lost
        allow_cache_up_to_level=allow_cache_up_to_level_,
        max_reduction_chunk_size=Vec3D(*max_reduction_chunk_size_),
        op=op_,
        bbox=bbox_,
        expand_bbox_resolution=expand_bbox_resolution,
        expand_bbox_backend=expand_bbox_backend,
        expand_bbox_processing=expand_bbox_processing,
        shrink_processing_chunk=shrink_processing_chunk,
        auto_divisibility=auto_divisibility,
        op_worker_type=op_worker_type,
        reduction_worker_type=reduction_worker_type,
        print_summary=print_summary,
        generate_ng_link=generate_ng_link,
        op_args=op_args,
        op_kwargs=op_kwargs_,
    )


def parse_bbox(
    dst: VolumetricBasedLayerProtocol | None,
    bbox: BBox3D | None,
    start_coord: Sequence[int] | None,
    end_coord: Sequence[int] | None,
    coord_resolution: Sequence | None,
    dst_resolution: Vec3D,
    auto_bbox: bool,
) -> BBox3D:

    if auto_bbox:
        if (
            start_coord is not None
            or end_coord is not None
            or coord_resolution is not None
            or bbox is not None
        ):
            raise ValueError(
                "`auto_bbox` was supplied, so `bbox`, `start_coord`, end_coord`, and"
                " `coord_resolution` cannot be specified."
            )
        assert dst is not None
        bbox_ = dst.backend.get_bounds(dst_resolution).bbox
    else:
        if bbox is None:
            if start_coord is None or end_coord is None or coord_resolution is None:
                raise ValueError(
                    "`bbox` and `auto_bbox` were not supplied, so `start_coord`, "
                    "end_coord`, and `coord_resolution` has to be specified."
                )
            bbox_ = BBox3D.from_coords(
                start_coord=start_coord, end_coord=end_coord, resolution=coord_resolution
            )
        else:
            if start_coord is not None or end_coord is not None or coord_resolution is not None:
                raise ValueError(
                    "`bbox` was supplied, so `start_coord`, end_coord`, and"
                    " `coord_resolution` cannot be specified."
                )
            bbox_ = bbox

    return bbox_


def _path_join_if_not_none(base: str | None, suffix: str) -> str | None:
    if base is None:
        return None
    else:
        return path.join(base, suffix)  # f"chunks_level_{level}"


def _expand_bbox_resolution(  # pylint: disable=line-too-long
    bbox: BBox3D,
    dst_resolution: Vec3D,
) -> BBox3D:
    bbox_new = bbox.snapped(Vec3D[float](0, 0, 0), dst_resolution, "expand")
    if bbox_new != bbox:
        logger.info(
            f"`expand_bbox_resolution` was set and the `bbox` was not integral in `dst_resolution`, "
            f"so the bbox has been modified: (in {dst_resolution.pformat()} {bbox.unit} pixels))\n"
            f"Received bbox:\t{bbox.pformat()} {bbox.unit}\n\t\t{bbox.pformat(dst_resolution)} px\n"
            f"\tshape:\t{(bbox.shape / dst_resolution).pformat()} px\n"
            f"New bbox:\t{bbox_new.pformat()} {bbox_new.unit}\n\t\t{bbox_new.pformat(dst_resolution)} px\n"
            f"\tshape:\t{(bbox_new.shape // dst_resolution).int().pformat()} px\n"
            f"Please note that this may affect chunk alignment requirements."
        )
    else:
        logger.info(
            "`expand_bbox_resolution` was set, but the `bbox` was already integral in `dst_resolution`, "
            "so no action has been taken."
        )

    return bbox_new


def _auto_divisibility(  # pylint: disable=line-too-long
    processing_chunk_sizes: Sequence[Vec3D[int]],
    processing_crop_pads: Sequence[Vec3D[int]],
    processing_blend_pads: Sequence[Vec3D[int]],
) -> Sequence[Vec3D[int]]:
    num_levels = len(processing_chunk_sizes)
    processing_chunk_sizes_new = list(deepcopy(processing_chunk_sizes))
    for level in range(0, num_levels - 1):
        i = num_levels - level - 1
        processing_chunk_size = processing_chunk_sizes_new[i]
        processing_chunk_size_higher = processing_chunk_sizes[i - 1]
        processing_blend_pad_higher = processing_blend_pads[i - 1]
        processing_crop_pad_higher = processing_crop_pads[i - 1]

        processing_region = (
            processing_chunk_size_higher
            + 2 * processing_crop_pad_higher
            + 2 * processing_blend_pad_higher
        )

        processing_chunk_sizes_new[i - 1] = Vec3D[int](
            *(
                processing_region // processing_chunk_size * processing_chunk_size
                - 2 * processing_crop_pad_higher
                - 2 * processing_blend_pad_higher
            )
        )

    # TODO FIX
    if processing_chunk_sizes_new != processing_chunk_sizes:
        logger.info(
            f"`auto_divisibility` was set and the divisibility requirements were not satisfied, "
            "so `processing_chunk_sizes` has been modified:\n"
            f"original:\t{', '.join(size.pformat() for size in processing_chunk_sizes)}\n"
            f"modified:\t{', '.join(size.pformat() for size in processing_chunk_sizes_new)}\n"
            f"Please note that this may affect the expected performance."
        )
    return processing_chunk_sizes_new


def _expand_bbox_backend(  # pylint: disable=line-too-long
    bbox: BBox3D,
    dst: VolumetricBasedLayerProtocol,
    dst_resolution: Vec3D,
) -> BBox3D:
    dst_backend_voxel_offset = dst.backend.get_voxel_offset(dst_resolution)
    dst_backend_chunk_size = dst.backend.get_chunk_size(dst_resolution)
    bbox_new = bbox.snapped(
        dst_backend_voxel_offset * dst_resolution,
        dst_backend_chunk_size * dst_resolution,
        "expand",
    )
    if bbox_new != bbox:
        logger.info(
            f"`expand_bbox_backend` was set and the `bbox` was not aligned to the `dst` layer's backend chunks, "
            f"so the bbox has been modified: (in {dst_resolution.pformat()} {bbox.unit} pixels))\n"
            f"`dst` voxel_offset:\t{dst_backend_voxel_offset.pformat()}\t chunk_size:\t"
            f"\t{dst_backend_chunk_size.pformat()}\n"
            f"Received bbox:\t{bbox.pformat()} {bbox.unit}\n\t\t{bbox.pformat(dst_resolution)} px\n"
            f"\tshape:\t{(bbox.shape / dst_resolution).pformat()} px\n"
            f"New bbox:\t{bbox_new.pformat()} {bbox_new.unit}\n\t\t{bbox_new.pformat(dst_resolution)} px\n"
            f"\tshape:\t{(bbox_new.shape // dst_resolution).int().pformat()} px\n"
            f"Please note that this may affect chunk alignment requirements."
        )
    else:
        logger.info(
            "`expand_bbox_backend` was set, but the `bbox` was already aligned to `dst` layer's backend chunks,  "
            "so no action has been taken."
        )
    return bbox_new


def _expand_bbox_processing(  # pylint: disable=line-too-long
    bbox: BBox3D,
    dst_resolution: Vec3D,
    processing_chunk_sizes: Sequence[Vec3D[int]],
    processing_gap: Vec3D[int],
) -> BBox3D:
    bbox_shape_in_res = round(bbox.shape / dst_resolution)
    bbox_shape_in_res_raw = round(bbox.shape / dst_resolution, VEC3D_PRECISION)
    if bbox_shape_in_res != bbox_shape_in_res_raw:
        raise ValueError(
            "To use `expand_bbox_processing`, the `bbox` must be integral in the "
            f"`dst_resolution`. Received {bbox.pformat()}, which is "
            f"{bbox.pformat(dst_resolution)} at the `dst_resolution` of "
            f"{dst_resolution.pformat()}. You may set `expand_bbox_resolution = True` to "
            "automatically expand the bbox to the nearest integral pixel."
        )
    bbox_old = bbox
    chunk_size_top = processing_chunk_sizes[0]
    translation_end = (chunk_size_top - bbox_shape_in_res) % (chunk_size_top + processing_gap)
    bbox = bbox.translated_end(translation_end, dst_resolution)
    if translation_end != Vec3D[int](0, 0, 0):
        logger.info(
            f"`expand_bbox_processing` was set and the `bbox` was not aligned to the top level "
            f"`processing_chunk_size` (with `processing_gap`s if applicable) in at least one dimension, "
            f"so the bbox has been modified: (in {dst_resolution.pformat()} {bbox_old.unit} pixels))\n"
            f"Received bbox:\t{bbox_old.pformat()} {bbox_old.unit}\n\t\t{bbox_old.pformat(dst_resolution)} px\n"
            f"\tshape:\t{(bbox_old.shape // dst_resolution).int().pformat()} px\n"
            f"New bbox:\t{bbox.pformat()} {bbox.unit}\n\t\t{bbox.pformat(dst_resolution)} px\n"
            f"\tshape:\t{(bbox.shape // dst_resolution).int().pformat()} px\n"
            f"Please note that this may affect chunk alignment requirements."
        )
    else:
        logger.info(
            "`expand_bbox_processing` was set, but the `bbox` was already aligned to processing chunks,  "
            "so no action has been taken."
        )
    return bbox


def _shrink_processing_chunk(  # pylint: disable=line-too-long
    bbox: BBox3D,
    dst_resolution: Vec3D,
    processing_chunk_sizes: Sequence[Vec3D[int]],
) -> Sequence[Vec3D[int]]:
    bbox_shape_in_res = round(bbox.shape / dst_resolution)
    bbox_shape_in_res_raw = round(bbox.shape / dst_resolution, VEC3D_PRECISION)
    if bbox_shape_in_res != bbox_shape_in_res_raw:
        raise ValueError(
            "To use `shrink_processing_chunk`, the `bbox` must be integral in the "
            f"`dst_resolution`. Received {bbox.pformat()}, which is "
            f"{bbox.pformat(dst_resolution)} at the `dst_resolution` of "
            f"{dst_resolution.pformat()}. You may set `expand_bbox_resolution = True` to "
            "automatically expand the bbox to the nearest integral pixel."
        )
    processing_chunk_sizes_old = deepcopy(processing_chunk_sizes)
    processing_chunk_sizes = list(processing_chunk_sizes)
    processing_chunk_sizes[0] = Vec3D[int](
        *[int(min(b, e)) for b, e in zip(bbox_shape_in_res, processing_chunk_sizes_old[0])]
    )
    for i in range(1, len(processing_chunk_sizes)):
        processing_chunk_sizes[i] = Vec3D[int](
            *[
                int(min(b, s))
                for b, s in zip(processing_chunk_sizes[i - 1], processing_chunk_sizes[i])
            ]
        )
    logger.info(
        f"`shrink_processing_chunk` was set, so the `processing_chunk_sizes` have been shrunken to fit the bbox where applicable.\n"
        f"Original `processing_chunk_sizes`:\t{', '.join([e.pformat() for e in processing_chunk_sizes_old])}\n"
        f"Shrunken `processing_chunk_sizes`:\t{', '.join([e.pformat() for e in processing_chunk_sizes])}\n"
        f"Please note that this may affect divisibility requirements."
    )
    return processing_chunk_sizes


def _make_ng_link(
    dst: VolumetricBasedLayerProtocol, bbox: BBox3D
) -> Optional[str]:  # pragma: no cover
    link_layers = []
    layer_strs = dst.pformat().split("\n")
    for layer_str in layer_strs:
        layer_name = layer_str.split(" ")[0]
        layer_path = layer_str.split(" ")[-1]
        try:
            build_cv_layer(layer_path)
            link_layers.append((layer_name, "image", "precomputed://" + layer_path.strip("/")))
        except FileNotFoundError:
            pass
    try:
        ng_link = make_ng_link(link_layers, bbox.start, print_to_logger=False)
    except RuntimeError:
        ng_link = make_ng_link(
            link_layers, bbox.start, print_to_logger=False, state_server_url=None
        )
    return ng_link


def _print_summary(  # pylint: disable=line-too-long, too-many-locals, too-many-statements, too-many-branches
    dst: VolumetricBasedLayerProtocol | None,
    dst_resolution: Vec3D,
    dst_tighten_bounds: bool,
    level_intermediaries_dirs: Sequence[str | None],
    skip_intermediaries: bool,
    processing_chunk_sizes: Sequence[Vec3D[int]],
    processing_blend_pads: Sequence[Vec3D[int]],
    processing_blend_modes: Sequence[Literal["linear", "quadratic", "max", "defer"]],
    processing_crop_pad: Vec3D[int],
    roi_crop_pads: Sequence[Vec3D[int]],
    max_reduction_chunk_size: Vec3D[int],
    allow_cache_up_to_level: int,
    bbox: BBox3D,
    original_bbox: BBox3D,
    expand_bbox_resolution: bool,
    expand_bbox_backend: bool,
    expand_bbox_processing: bool,
    num_levels: int,
    num_chunks: Sequence[int],
    use_checkerboard: Sequence[bool],
    op_name: str,
    generate_ng_link: bool,
    op_args: Iterable,
    op_kwargs: Mapping[str, Any],
    processing_gap: Vec3D[int],
) -> None:  # pragma: no cover
    summary = ""
    summary += (
        lrpad("  SubchunkableApplyFlow Parameter Summary  ", bounds="+", filler="=", length=120)
        + "\n"
    )
    summary += lrpad(length=120) + "\n"
    summary += lrpad(f"Generated {utcnow_ISO8601()}", 1, length=120) + "\n"
    summary += lrpad(length=120) + "\n"
    summary += lrpad(" Dataset Information ", bounds="|", filler="=", length=120) + "\n"
    summary += lrpad(length=120) + "\n"
    original_vol = 0.0
    if original_bbox != bbox:
        summary += (
            lrpad("Original BBox bounds, before requested modifications:", 1, length=120) + "\n"
        )
        if dst_tighten_bounds:
            summary += (
                lrpad(
                    "(Output(s) will be tightened to this BBox, after expansion to resolution if requested)",
                    2,
                    length=120,
                )
                + "\n"
            )
        summary += lrpad(f"in {original_bbox.unit}:", 2, length=120) + "\n"
        summary += lrpad(f"{original_bbox.pformat()} {original_bbox.unit}", 3, length=120) + "\n"
        summary += (
            lrpad(f"shape: {original_bbox.shape.pformat()} {original_bbox.unit}", 3, length=120)
            + "\n"
        )
        summary += lrpad(f"in {dst_resolution.pformat()} nm voxels:", 2, length=120) + "\n"
        summary += lrpad(f"{original_bbox.pformat(dst_resolution)} px", 3, length=120) + "\n"
        summary += (
            lrpad(
                f"shape: {(original_bbox.shape // dst_resolution).int().pformat()} px",
                3,
                length=120,
            )
            + "\n"
        )
        summary += lrpad(length=120) + "\n"
        original_vol = original_bbox.get_size()
        if original_vol < 1e18:
            summary += (
                lrpad(f"Volume of ROI: {original_vol*1e-9:10.3f} um^3", 1, length=120) + "\n"
            )
        else:
            summary += (
                lrpad(f"Volume of ROI: {original_vol*1e-18:10.3f} mm^3", 1, length=120) + "\n"
            )
        summary += lrpad(length=120) + "\n"

        summary += lrpad("Requested modifications to the BBox:", 1, length=120) + "\n"
        if expand_bbox_resolution:
            summary += (
                lrpad(
                    "`expand_bbox_resolution`: expand to be integral in the resolution provided",
                    2,
                    length=120,
                )
                + "\n"
            )
        if expand_bbox_backend:
            summary += (
                lrpad(
                    "`expand_bbox_backend`: expand to align to the backend chunks", 2, length=120
                )
                + "\n"
            )
        if expand_bbox_processing:
            summary += (
                lrpad(
                    "`expand_bbox_processing`: expand to align to the top level processing chunks",
                    2,
                    length=120,
                )
                + "\n"
            )
        summary += lrpad(length=120) + "\n"

    summary += lrpad("BBox bounds for processing:", 1, length=120) + "\n"
    summary += lrpad(f"in {bbox.unit}:", 2, length=120) + "\n"
    summary += lrpad(f"{bbox.pformat()} {bbox.unit}", 3, length=120) + "\n"
    summary += lrpad(f"shape: {bbox.shape.pformat()} {bbox.unit}", 3, length=120) + "\n"
    summary += lrpad(f"in {dst_resolution.pformat()} nm voxels:", 2, length=120) + "\n"
    summary += lrpad(f"{bbox.pformat(dst_resolution)} px", 3, length=120) + "\n"
    summary += (
        lrpad(f"shape: {(bbox.shape // dst_resolution).int().pformat()} px", 3, length=120) + "\n"
    )
    summary += lrpad(length=120) + "\n"
    vol = bbox.get_size()
    if vol < 1e18:
        summary += lrpad(f"Volume of ROI: {vol*1e-9:10.3f} um^3", 1, length=120) + "\n"
    else:
        summary += lrpad(f"Volume of ROI: {vol*1e-18:10.3f} mm^3", 1, length=120) + "\n"
    summary += lrpad(length=120) + "\n"
    if original_vol != 0.0:
        summary += (
            lrpad(f"Overhead from BBox modification: {vol/original_vol:10.3f} X", 1, length=120)
            + "\n"
        )
        summary += lrpad(length=120) + "\n"
    summary += lrpad("Output location(s):", 1, length=120) + "\n"
    if dst is None:
        summary += lrpad("`dst` is None\n")
        layer_strs = []
    else:
        layer_strs = dst.pformat().split("\n")
    for layer_str in layer_strs:
        summary += lrpad(layer_str, 2, length=120) + "\n"
    if generate_ng_link and dst is not None:
        ng_link = _make_ng_link(dst, bbox)
        if ng_link is not None:
            summary += lrpad(length=120) + "\n"
            summary += lrpad("Neuroglancer link:", length=120) + "\n"
            summary += lrpad(f"{ng_link}", 2, length=120) + "\n"
    summary += lrpad(length=120) + "\n"
    summary += lrpad(" Operation Information ", bounds="|", filler="=", length=120) + "\n"
    summary += lrpad("", length=120) + "\n"
    summary += lrpad(f"Operation: {op_name}  ", length=120) + "\n"
    summary += (
        lrpad(
            f"Post-pad input size:     {(processing_chunk_sizes[-1] + 2*processing_crop_pad + 2*processing_blend_pads[-1]).pformat()}  ",
            level=2,
            length=120,
        )
        + "\n"
    )
    summary += (
        lrpad(
            f"Pre-pad chunk size:  {processing_chunk_sizes[-1].pformat()}  ", level=3, length=120
        )
        + "\n"
    )
    summary += (
        lrpad(f"Crop pad:            {processing_crop_pad.pformat()}  ", level=3, length=120)
        + "\n"
    )
    summary += (
        lrpad(f"Blend pad:           {processing_blend_pads[-1].pformat()}  ", level=3, length=120)
        + "\n"
    )
    summary += lrpad(f"Processing gap: {processing_gap.pformat()}  ", level=2, length=120) + "\n"

    summary += lrpad(f"# of op_args supplied: {len(list(op_args))}", level=2, length=120) + "\n"
    summary += lrpad("op_kwargs supplied:", level=2, length=120) + "\n"
    for k, v in op_kwargs.items():
        summary += (
            lrpad(
                f"{lrpad(f'{k}:', level=0, length = 15, bounds = '')}" f"{type(v).__name__}",
                level=3,
                length=120,
            )
            + "\n"
        )

    summary += lrpad(length=120) + "\n"
    summary += lrpad(" Subchunking Information ", bounds="|", filler="=", length=120) + "\n"
    summary += lrpad(length=120) + "\n"
    summary += (
        lrpad(
            " Level  # of chunks   Processing chunk size  ROI crop pad"
            "    Blend pad       Blend mode    Checkerboard  Cached",
            length=120,
        )
        + "\n"
    )
    for level in range(num_levels - 1, -1, -1):
        i = num_levels - level - 1
        summary += (
            lrpad(
                f" {level}      "
                f"{lrpad(str(num_chunks[i]), level = 0, length = 10, bounds ='')}     "
                f"{lrpad(processing_chunk_sizes[i].pformat(), level = 0, length = 22, bounds = '')}  "
                f"{lrpad(roi_crop_pads[i].pformat(), level = 0, length = 15, bounds = '')}  "
                f"{lrpad(processing_blend_pads[i].pformat(), level = 0, length = 15, bounds = '')}  "
                f"{lrpad(processing_blend_modes[i], level = 0, length = 13, bounds = '')}  "
                f"{lrpad(str(use_checkerboard[i]), level = 0, length = 15, bounds = '')}"
                f"{lrpad(str(level <= allow_cache_up_to_level), level = 0, length = 10, bounds = '')}",
                length=120,
            )
            + "\n"
        )
    if not skip_intermediaries:
        summary += lrpad(length=120) + "\n"
        summary += (
            lrpad(
                " Level  Max red. chunk size  Intermediary dir",
                length=120,
            )
            + "\n"
        )
        for level in range(num_levels - 1, -1, -1):
            i = num_levels - level - 1
            summary += (
                lrpad(
                    f" {level}      "
                    f"{lrpad(max_reduction_chunk_size.pformat(), level = 0, length = 22, bounds = '')}"
                    f"{level_intermediaries_dirs[i]}",
                    length=120,
                )
                + "\n"
            )
    summary += lrpad("", length=120) + "\n"
    summary += lrpad("", bounds="+", filler="=", length=120)
    logger.info(summary)


def _build_subchunkable_apply_flow(  # pylint: disable=keyword-arg-before-vararg, line-too-long, too-many-locals, too-many-branches, too-many-statements
    dst: VolumetricBasedLayerProtocol | None,
    dst_resolution: Vec3D,
    dst_tighten_bounds: bool,
    level_intermediaries_dirs: Sequence[str | None],
    skip_intermediaries: bool,
    processing_chunk_sizes: Sequence[Vec3D[int]],
    processing_crop_pads: Sequence[Vec3D[int]],
    processing_blend_pads: Sequence[Vec3D[int]],
    processing_blend_modes: Sequence[Literal["linear", "quadratic", "max", "defer"]],
    max_reduction_chunk_size: Vec3D[int],
    allow_cache_up_to_level: int,
    bbox: BBox3D,
    expand_bbox_resolution: bool,
    expand_bbox_backend: bool,
    expand_bbox_processing: bool,
    shrink_processing_chunk: bool,
    auto_divisibility: bool,
    op_worker_type: str | None,
    reduction_worker_type: str | None,
    print_summary: bool,
    generate_ng_link: bool,
    op: VolumetricOpProtocol[P, None, Any],
    op_args: P.args,
    op_kwargs: P.kwargs,
    processing_gap: Vec3D[int],
) -> mazepa.Flow:
    num_levels = len(processing_chunk_sizes)

    if auto_divisibility:
        processing_chunk_sizes = _auto_divisibility(
            processing_chunk_sizes, processing_crop_pads, processing_blend_pads
        )

    original_bbox = deepcopy(bbox)

    if expand_bbox_resolution:
        bbox = _expand_bbox_resolution(bbox, dst_resolution)

    original_bbox_resolution = deepcopy(bbox)

    if expand_bbox_backend and dst is not None:
        bbox = _expand_bbox_backend(bbox, dst, dst_resolution)

    if shrink_processing_chunk:
        processing_chunk_sizes = _shrink_processing_chunk(
            bbox, dst_resolution, processing_chunk_sizes
        )
    elif expand_bbox_processing:
        bbox = _expand_bbox_processing(
            bbox, dst_resolution, processing_chunk_sizes, processing_gap
        )

    idx = VolumetricIndex(resolution=dst_resolution, bbox=bbox)
    level0_op = op.with_added_crop_pad(processing_crop_pads[-1])
    if hasattr(level0_op, "get_operation_name"):
        op_name = level0_op.get_operation_name()
    else:  # pragma: no cover # TODO: make @property def name part of the protocol
        op_name = type(level0_op).__name__

    """
    Check that the sizes are correct. Note that the the order has to be reversed for indexing.
    At level l (where processing_chunk_sizes[num_levels] = idx.stop-idx.start):
    processing_chunk_sizes[l+1] + 2*processing_crop_pads[l+1] + 2*processing_blend_pads[l+1] % processing_chunk_sizes[l] == 0
    processing_blend_pads[l] * 2 <= processing_chunk_sizes[l]
    """
    logger.info("Checking the arguments given.")
    num_chunks = []
    for level in range(num_levels - 1, -1, -1):
        i = num_levels - level - 1
        if i == 0:
            processing_chunk_size_higher = idx.shape
            processing_blend_pad_higher = Vec3D[int](0, 0, 0)
            processing_crop_pad_higher = Vec3D[int](0, 0, 0)
            processing_gap_higher = processing_gap
        else:
            processing_chunk_size_higher = processing_chunk_sizes[i - 1]
            processing_blend_pad_higher = processing_blend_pads[i - 1]
            processing_crop_pad_higher = processing_crop_pads[i - 1]
            processing_gap_higher = Vec3D[int](0, 0, 0)
        processing_chunk_size = processing_chunk_sizes[i]
        processing_blend_pad = processing_blend_pads[i]

        processing_region = (
            processing_chunk_size_higher
            + 2 * processing_crop_pad_higher
            + 2 * processing_blend_pad_higher
        )

        n_region_chunks = (processing_region + processing_gap_higher) / (
            processing_chunk_size + processing_gap_higher
        )
        n_region_chunks_rounded = Vec3D[int](*(max(1, round(e)) for e in n_region_chunks))
        num_chunks.append(math.prod(n_region_chunks_rounded))
        if n_region_chunks != n_region_chunks_rounded:
            if (processing_region + processing_gap_higher) % n_region_chunks_rounded == Vec3D(
                0, 0, 0
            ):
                rec_processing_chunk_size = (
                    (processing_region + processing_gap) / n_region_chunks_rounded
                ).int()
                rec_str = f"Recommendation for `processing_chunk_size[level]`:\t\t\t\t{rec_processing_chunk_size}"
            else:
                rec_str = (
                    "Unable to recommend processing_chunk_size[level]: try using a more divisible"
                    " `processing_crop_pad[level+1]` and/or `processing_blend_pad[level+1]`."
                )
            error_str = (
                "At each level (where the 0-th level is the smallest), the"
                " `processing_chunk_size[level+1]` + 2*`processing_crop_pad[level+1]` + 2*`processing_blend_pad[level+1]`"
                " + processing_gap must be"
                f" evenly divisible by the `processing_chunk_size[level]` + processing_gap (processing_gap applies only on top level).\n"
                f"\nAt level {level}, received:\n"
                f"`processing_chunk_size[level+1]`:\t\t\t\t\t\t{processing_chunk_size_higher}\n"
                f"`applicable processing_gap`:\t\t\t\t\t\t\t\t{processing_gap_higher}\n"
                f"`processing_crop_pad[level+1]` ((0, 0, 0) for the top level):\t\t\t{processing_crop_pad_higher}\n"
                f"`processing_blend_pad[level+1]`:\t\t\t\t\t\t{processing_blend_pad_higher}\n"
                f"Size of the region to be processed for the level:\t\t\t\t{processing_region}\n"
                f"Which must be (but is not) divisible by: `processing_chunk_size[level]`:\t{processing_chunk_size}\n\n"
            )
            raise ValueError(error_str + rec_str)
        if not processing_blend_pad * 2 <= processing_chunk_size:
            raise ValueError(
                "At each level (where the 0-th level is the smallest), the `processing_blend_pad[level]` must"
                f" be at most half of the `processing_chunk_size[level]`. At level {level}, received:\n"
                f"`processing_blend_pad[level]`: {processing_blend_pad}\n"
                f"`processing_chunk_size[level]`: {processing_chunk_size}"
            )
    num_chunks_below = deepcopy(num_chunks)
    num_chunks_below.append(1)
    for i in range(1, num_levels):
        num_chunks[i] *= num_chunks[i - 1]
    for i in range(num_levels - 1, -1, -1):
        num_chunks_below[i] *= num_chunks_below[i + 1]

    """
    Append the zero roi_crop pad for the top level, and remove the bottom level baked into the operation
    """
    roi_crop_pads = [Vec3D[int](0, 0, 0)] + list(processing_crop_pads[:-1])

    """
    Check for no checkerboarding and tighten bounds if specifed
    """
    use_checkerboard = []
    for i in range(0, num_levels - 1):
        if roi_crop_pads[i] == Vec3D[int](0, 0, 0) and processing_blend_pads[i] == Vec3D[int](
            0, 0, 0
        ):
            logger.info(
                f"Level {num_levels - i - 1}: Chunks will not be checkerboarded since `processing_crop_pad[level+1]`"
                f" (always {Vec3D[int](0, 0, 0)} for the top level) and `processing_blend_pad[level]` are both {Vec3D[int](0, 0, 0)}."
            )
            use_checkerboard.append(False)
        else:
            use_checkerboard.append(True)
    if processing_blend_pads[num_levels - 1] == Vec3D[int](0, 0, 0):
        logger.info(
            f"Level 0: Chunks will not be checkerboarded since `processing_blend_pad[level]` is {Vec3D[int](0, 0, 0)}."
        )
        use_checkerboard.append(False)
    else:
        use_checkerboard.append(True)
    if dst is None:
        logger.info("`dst` is None; skipping generation of standard output.")
    else:
        if skip_intermediaries is True:
            logger.info(
                "Since intermediaries are skipped, the ROI and any writing done to the final destination are "
                "required to be chunk-aligned."
            )
            dst = deepcopy(dst).with_changes(
                backend=dst.backend.with_changes(enforce_chunk_aligned_writes=True)
            )
        else:
            dst = deepcopy(dst).with_changes(
                backend=dst.backend.with_changes(enforce_chunk_aligned_writes=False)
            )
        if dst_tighten_bounds is True:
            dst = deepcopy(dst).with_changes(
                backend=dst.backend.with_changes(
                    voxel_offset_res=(
                        (original_bbox_resolution.start / dst_resolution).int(),
                        dst_resolution,
                    ),
                    dataset_size_res=(
                        (original_bbox_resolution.shape / dst_resolution).int(),
                        dst_resolution,
                    ),
                )
            )

    if print_summary:
        _print_summary(
            dst=dst,
            dst_resolution=dst_resolution,
            dst_tighten_bounds=dst_tighten_bounds,
            level_intermediaries_dirs=level_intermediaries_dirs,
            skip_intermediaries=skip_intermediaries,
            processing_chunk_sizes=processing_chunk_sizes,
            processing_blend_pads=processing_blend_pads,
            processing_blend_modes=processing_blend_modes,
            processing_crop_pad=processing_crop_pads[-1],
            roi_crop_pads=roi_crop_pads,
            max_reduction_chunk_size=max_reduction_chunk_size,
            allow_cache_up_to_level=allow_cache_up_to_level,
            bbox=bbox,
            original_bbox=original_bbox,
            expand_bbox_resolution=expand_bbox_resolution,
            expand_bbox_backend=expand_bbox_resolution,
            expand_bbox_processing=expand_bbox_processing,
            num_levels=num_levels,
            num_chunks=num_chunks,
            use_checkerboard=use_checkerboard,
            generate_ng_link=generate_ng_link,
            op_name=op_name,
            op_args=op_args,
            op_kwargs=op_kwargs,
            processing_gap=processing_gap,
        )
    """
    Generate flow id for deconflicting intermediaries - must be deterministic for proper
    identification across Python sessions
    """
    flow_id = id_generation.generate_invocation_id(kwargs=locals(), prefix="subchunkable")

    """
    Basic building blocks where the work gets done, at the very bottom
    """
    flow_schema: VolumetricApplyFlowSchema = VolumetricApplyFlowSchema(
        op=level0_op,
        processing_chunk_size=processing_chunk_sizes[-1],
        max_reduction_chunk_size=max_reduction_chunk_size,
        dst_resolution=dst_resolution,
        roi_crop_pad=roi_crop_pads[-1],
        processing_blend_pad=processing_blend_pads[-1],
        processing_blend_mode=processing_blend_modes[-1],
        processing_gap=None,
        intermediaries_dir=_path_join_if_not_none(level_intermediaries_dirs[-1], "chunks_level_0"),
        allow_cache=(allow_cache_up_to_level >= 1),
        clear_cache_on_return=(allow_cache_up_to_level == 1),
        force_intermediaries=not (skip_intermediaries),
        flow_id=flow_id,
        l0_chunks_per_task=num_chunks_below[-1],
        op_worker_type=op_worker_type,
        reduction_worker_type=reduction_worker_type,
    )
    """
    Iteratively build the hierarchy of schemas
    """
    for level in range(1, num_levels):
        flow_schema = VolumetricApplyFlowSchema(
            op=DelegatedSubchunkedOperation(  # type:ignore #readability over typing here
                flow_schema,
                op_name,
                level,
            ),
            processing_chunk_size=processing_chunk_sizes[-level - 1],
            max_reduction_chunk_size=max_reduction_chunk_size,
            dst_resolution=dst_resolution,
            roi_crop_pad=roi_crop_pads[-level - 1],
            processing_blend_pad=processing_blend_pads[-level - 1],
            processing_blend_mode=processing_blend_modes[-level - 1],
            processing_gap=processing_gap if level == num_levels - 1 else None,
            intermediaries_dir=_path_join_if_not_none(
                level_intermediaries_dirs[-level - 1], f"chunks_level_{level}"
            ),
            allow_cache=(allow_cache_up_to_level >= level + 1),
            clear_cache_on_return=(allow_cache_up_to_level == level + 1),
            force_intermediaries=not (skip_intermediaries),
            flow_id=flow_id,
            l0_chunks_per_task=num_chunks_below[-level - 1],
            op_worker_type=op_worker_type,
            reduction_worker_type=reduction_worker_type,
        )
    return flow_schema(idx, dst, op_args, op_kwargs)
