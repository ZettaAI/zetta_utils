"""
Builder support for VolumetricAnnotationLayer and related classes.

Usage Examples

  CUE Specification:
  annotation_layer: {
      "@type": "build_annotation_layer"
      path: "gs://bucket/annotations"
      mode: "write"
      resolution: [4, 4, 40]
      dataset_size: [1000, 1000, 100]
      voxel_offset: [0, 0, 0]
      property_specs: [
          {
              "@type": "build_property_spec"
              id: "score"
              type: "float32"
              description: "Confidence score"
          }
      ]
      relationships: [
          {
              "@type": "build_relationship"
              id: "presyn_cell"
          }
      ]
  }

  Direct Python Usage:
  from zetta_utils.layer.volumetric.annotation.build import build_annotation_layer
  from zetta_utils.layer.volumetric.annotation.annotations import PropertySpec, Relationship

  layer = build_annotation_layer(
      path="/path/to/annotations",
      property_specs=[PropertySpec(id="score", type="float32")],
      relationships=[Relationship(id="presyn_cell")],
      # ... other parameters
  )
"""
from __future__ import annotations

from typing import Iterable, Literal, Sequence

from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric.annotation.annotations import (
    PropertySpec,
    Relationship,
)
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
@builder.register("build_property_spec")
def build_property_spec(
    id: str,  # pylint: disable=redefined-builtin
    type: str,  # pylint: disable=redefined-builtin
    description: str | None = None,
    enum_values: Sequence[int | float] | None = None,
    enum_labels: Sequence[str] | None = None,
) -> PropertySpec:
    """Build a PropertySpec for annotation properties.

    :param id: Property identifier (must match pattern: ^[a-z][a-zA-Z0-9_]*$)
    :param type: Property type (rgb, rgba, uint8, int8, uint16, int16, uint32, int32, or float32)
    :param description: Optional description of the property
    :param enum_values: Optional list of numeric values for enumerated properties
    :param enum_labels: Optional list of string labels corresponding to enum values
    :return: PropertySpec instance
    """
    return PropertySpec(
        id=id,
        type=type,
        description=description,
        enum_values=list(enum_values) if enum_values is not None else None,
        enum_labels=list(enum_labels) if enum_labels is not None else None,
    )


@typechecked
@builder.register("build_relationship")
def build_relationship(
    id: str,  # pylint: disable=redefined-builtin
    key: str | None = None,
) -> Relationship:
    """Build a Relationship for annotation relationships.

    :param id: Relationship identifier
    :param key: Optional directory key (auto-generated from id if not provided)
    :return: Relationship instance
    """
    return Relationship(id=id, key=key)


@typechecked
@builder.register("build_annotation_layer")
def build_annotation_layer(  # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    path: str,
    resolution: Sequence[float] | None = None,
    dataset_size: Sequence[int] | None = None,
    voxel_offset: Sequence[int] | None = None,
    index: VolumetricIndex | None = None,
    bbox: BBox3D | None = None,
    chunk_sizes: Sequence[Sequence[int]] | None = None,
    mode: Literal["read", "write", "replace", "update"] = "write",
    annotation_type: Literal["POINT", "LINE"] | None = None,
    default_desired_resolution: Sequence[float] | None = None,
    index_resolution: Sequence[float] | None = None,
    allow_slice_rounding: bool = False,
    index_procs: Iterable[IndexProcessor[VolumetricIndex]] = (),
    read_procs: Iterable[AnnotationDataProcT] = (),
    write_procs: Iterable[AnnotationDataWriteProcT] = (),
    property_specs: Iterable[PropertySpec] = (),
    relationships: Iterable[Relationship] = (),
) -> VolumetricAnnotationLayer:
    """Build an AnnotationLayer (spatially indexed annotations in precomputed file format).

    :param path: Path to the precomputed file (directory).
    :param resolution: (x, y, z) size of one voxel, in nm.
    :param dataset_size: Precomputed dataset size (in voxels) for all scales.
    :param voxel_offset: start of the dataset volume (in voxels) for all scales.
    :param index: VolumetricIndex indicating dataset size and resolution. Note that
      for new files, you must provide either (resolution, dataset_size, voxel_offset),
      (bbox, resolution), or index, but not multiple combinations. For existing files,
      all these are optional.
    :param bbox: BBox3D indicating the spatial bounds. If provided with resolution,
      a VolumetricIndex will be constructed from these.
    :param chunk_sizes: Chunk sizes for spatial index; defaults to a single chunk for
      new files (or the existing chunk structure for existing files).
    :param mode: How the file should be created or opened:
       "read": for reading only; throws error if file does not exist.
       "write": for writing; throws error if file exists.
       "replace": for writing; if file exists, it is cleared of all data.
       "update": for writing additional data; throws error if file does not exist.
    :param annotation_type: Type of annotations (POINT or LINE).
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
    :param property_specs: List of PropertySpec objects defining annotation properties.
    :param relationships: List of Relationship objects defining annotation relationships.
    :return: Layer built according to the spec.
    """
    if "/|neuroglancer-precomputed:" in path:
        path = path.split("/|neuroglancer-precomputed:")[0]
    (
        dims,
        lower_bound,
        upper_bound,
        anno_type,
        spatial_entries,
        existing_props,
        existing_rels,
    ) = read_info(path)
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
        if annotation_type and annotation_type != anno_type:
            raise IOError(
                f"Given annotation_type {annotation_type} "
                "does not match existing file type {anno_type}"
            )  # pragma: no cover
        annotation_type = anno_type

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
            # Check for bbox + resolution combination
            if bbox is not None and resolution is not None:
                if dataset_size is not None or voxel_offset is not None:
                    raise ValueError(
                        "when `bbox` and `resolution` are provided, `dataset_size` and "
                        "`voxel_offset` should not be provided"
                    )
                if len(resolution) != 3:
                    raise ValueError(f"`resolution` needs 3 elements, not {len(resolution)}")
                index = VolumetricIndex(bbox=bbox, resolution=Vec3D(*resolution))
            # Check for resolution + dataset_size + voxel_offset combination
            elif bbox is None:
                if resolution is None:
                    raise ValueError(
                        "when `index` is not provided, either (`bbox` and `resolution`) or "
                        "(`resolution`, `dataset_size`, and `voxel_offset`) are required"
                    )
                if dataset_size is None:
                    raise ValueError("when `index` is not provided, `dataset_size` is required")
                if voxel_offset is None:
                    raise ValueError("when `index` is not provided, `voxel_offset` is required")
                if len(resolution) != 3:
                    raise ValueError(f"`resolution` needs 3 elements, not {len(resolution)}")
                if len(dataset_size) != 3:
                    raise ValueError(f"`dataset_size` needs 3 elements, not {len(dataset_size)}")
                if len(voxel_offset) != 3:
                    raise ValueError(f"`voxel_offset` needs 3 elements, not {len(voxel_offset)}")
                end_coord = tuple(a + b for a, b in zip(voxel_offset, dataset_size))
                index = VolumetricIndex.from_coords(voxel_offset, end_coord, resolution)
            else:
                raise ValueError("when `bbox` is provided, `resolution` is also required")
        else:
            index = file_index
    assert index is not None

    if mode in ("read", "update"):
        assert file_chunk_sizes
        chunk_sizes = file_chunk_sizes
    else:
        if chunk_sizes is None:
            chunk_sizes = []

    # Use existing properties and relationships if file exists and none provided
    final_property_specs = list(property_specs) if property_specs else (existing_props or [])
    final_relationships = list(relationships) if relationships else (existing_rels or [])

    backend = AnnotationLayerBackend(
        path=path,
        index=index,
        annotation_type=annotation_type,
        chunk_sizes=chunk_sizes,
        property_specs=final_property_specs,
        relationships=final_relationships,
    )
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
