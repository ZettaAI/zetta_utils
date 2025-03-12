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
    resolution: Sequence[float] | None = None,
    dataset_size: Sequence[int] | None = None,
    voxel_offset: Sequence[int] | None = None,
    index: VolumetricIndex | None = None,
    chunk_sizes: Sequence[Sequence[int]] | None = None,
    mode: Literal["read", "write", "replace", "update"] = "write",
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
    if "/|neuroglancer-precomputed:" in path:
        path = path.split("/|neuroglancer-precomputed:")[0]
    dims, lower_bound, upper_bound, spatial_entries = read_info(path)
    file_exists = spatial_entries is not None
    file_resolution: list[float] = []
    file_index = None
    file_chunk_sizes = []
    if file_exists:
        for i in [0, 1, 2]:
            numAndUnit = dims["xyz"[i]]
            assert numAndUnit[1] == "nm", "Only dimensions in 'nm' are supported for reading"
            file_resolution.append(numAndUnit[0])

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

    if index is None:
        if mode == "write" or (mode == "replace" and not file_exists):
            if resolution is None:
                raise ValueError("when `index` is not provided, `resolution` is required")
            if dataset_size is None:
                raise ValueError("when `index` is not provided, `dataset_size` is required")
            if voxel_offset is None:
                raise ValueError("when `index` is not provided, `voxel_offset` is required")
            if len(resolution) != 3:
                raise ValueError(f"`resolution` needs 3 elements, not {len(resolution)}")
            if len(dataset_size) != 3:
                raise ValueError(f"`dataset_size` needs 3 elements, not {len(dataset_size)}")
            if len(voxel_offset) != 3:
                raise ValueError(f"`dataset_size` needs 3 elements, not {len(voxel_offset)}")
            end_coord = tuple(a + b for a, b in zip(voxel_offset, dataset_size))
            index = VolumetricIndex.from_coords(voxel_offset, end_coord, resolution)
        else:
            index = file_index
    assert index is not None

    if mode in ("read", "update"):
        assert file_chunk_sizes
        chunk_sizes = file_chunk_sizes
    else:
        if chunk_sizes is None:
            chunk_sizes = []

    backend = AnnotationLayerBackend(path=path, index=index, chunk_sizes=chunk_sizes)
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
