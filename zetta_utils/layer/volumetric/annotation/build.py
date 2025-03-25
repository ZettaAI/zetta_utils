from __future__ import annotations

from typing import Iterable, Literal, Sequence

from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.geometry import Vec3D
from zetta_utils.layer.volumetric.annotation.backend import (
    AnnotationLayerBackend,
    read_info,
)
from zetta_utils.layer.volumetric.annotation.layer import (
    AnnotationDataProcT,
    AnnotationDataWriteProcT,
    VolumetricAnnotationLayer,
)
from zetta_utils.layer.volumetric.index import VolumetricIndex

from ... import IndexProcessor


@typechecked
@builder.register("build_annotation_layer")
def build_annotation_layer(  # pylint: disable=too-many-locals, too-many-branches
    path: str,
    info_resolution: Sequence[float] | None = None,
    info_bbox: BBox3D | None = None,
    chunk_sizes: Sequence[Sequence[int]] | None = None,
    mode: Literal["read", "write", "replace", "update"] = "read",
    default_desired_resolution: Sequence[float] | None = None,
    index_resolution: Sequence[float] | None = None,
    allow_slice_rounding: bool = False,
    index_procs: Iterable[IndexProcessor[VolumetricIndex]] = (),
    read_procs: Iterable[AnnotationDataProcT] = (),
    write_procs: Iterable[AnnotationDataWriteProcT] = (),
) -> VolumetricAnnotationLayer:
    """Build an AnnotationLayer (spatially indexed annotations in precomputed file format).

    :param path: Path to the precomputed file (directory).
    :param resolution: (x, y, z) size of one voxel, in nm.
    :param dataset_size: Precomputed dataset size (in voxels) for all scales.
    :param voxel_offset: start of the dataset volume (in voxels) for all scales.
    :param index: VolumetricIndex indicating dataset size and resolution. Note that
      for new files, you must provide either (resolution, dataset_size, voxel_offset)
      or index, but not both. For existing files, all these are optional.
    :param chunk_sizes: Chunk sizes for spatial index; defaults to a single chunk for
      new files (or the existing chunk structure for existing files).
    :param mode: How the file should be created or opened:
       "read": for reading only; throws error if file does not exist.
       "write": for writing; throws error if file exists.
       "replace": for writing; if file exists, it is cleared of all data.
       "update": for writing additional data; throws error if file does not exist.
    :param readonly: Whether the layer is read-only.
    :param default_desired_resolution: Default resolution to use for reading data.
    :param index_resolution: Resolution to use for indexing.
    :param allow_slice_rounding: Whether to allow rounding of slice indices.
    :param index_procs: List of processors that will be applied to the index
        prior to IO operations.
    :param read_procs: List of processors that will be applied to the read data before
        returning it to the user.
    :param write_procs: List of processors that will be applied to the data given by
        the user before writing it to the backend.
    :return: Layer built according to the spec.
    """
    dims, lower_bound, upper_bound, spatial_entries = read_info(path)
    file_exists = spatial_entries is not None
    file_resolution: list[float] = []
    file_index = None
    file_chunk_sizes = []
    if file_exists:
        for i in [0, 1, 2]:
            num_and_unit = dims["xyz"[i]]
            if num_and_unit[1] == "m":
                file_resolution.append(num_and_unit[0] * 1e9)
            elif num_and_unit[1] == "nm":
                file_resolution.append(num_and_unit[0])
            else:
                raise ValueError(
                    f"Only dimensions in 'nm' or 'm' are supported, got '{num_and_unit[1]}'"
                )

        file_index = VolumetricIndex.from_coords(
            lower_bound,
            upper_bound,
            Vec3D(file_resolution[0], file_resolution[1], file_resolution[2]),
        )
        file_chunk_sizes = [se.chunk_size for se in spatial_entries]

    if mode in ("read", "update") and not file_exists:
        raise IOError(
            f"AnnotationLayer built with mode {mode}, but file does not exist (path: {path})"
        )
    if mode == "write" and file_exists:
        raise IOError(
            f"AnnotationLayer built with mode {mode}, but file already exists (path: {path})"
        )

    if mode == "write" or (mode == "replace" and not file_exists):
        if info_resolution is None:
            raise ValueError("when `mode` is `write` or `replace`, `info_resolution` is required")
        if info_bbox is None:
            raise ValueError("when `mode` is `write` or `replace`, `info_bbox` is required")
        if len(info_resolution) != 3:
            raise ValueError(f"`resolution` needs 3 elements, not {len(info_resolution)}")

        index = VolumetricIndex(resolution=Vec3D(*info_resolution), bbox=info_bbox)
    else:
        index = file_index

    if mode in ("read", "update"):
        assert file_chunk_sizes
        chunk_sizes = file_chunk_sizes
    else:
        if chunk_sizes is None:
            chunk_sizes = []
    backend = AnnotationLayerBackend(path=path, index=index, chunk_sizes=chunk_sizes)
    if mode in ("write", "replace"):
        backend.write_info_file()

    if mode == "replace":
        backend.clear()

    # Now create the VolumetricAnnotationLayer with the backend
    result = VolumetricAnnotationLayer(
        backend=backend,
        readonly=mode == "read",
        index_resolution=Vec3D(*index_resolution) if index_resolution else None,
        default_desired_resolution=(
            Vec3D(*default_desired_resolution) if default_desired_resolution else None
        ),
        allow_slice_rounding=allow_slice_rounding,
        index_procs=tuple(index_procs),
        read_procs=tuple(read_procs),
        write_procs=tuple(write_procs),
    )
    return result
