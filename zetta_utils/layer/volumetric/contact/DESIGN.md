# Contact Layer Format Design

A chunked spatial storage format for contact data between segmentation objects.

## Overview

Contacts represent interfaces between two segments. Each contact has:
- A unique integer ID
- A center of mass (COM) in 3D space
- Contact faces (3D points with affinity values)
- Optional local point clouds (mesh samples around COM)
- Optional merge decisions from various authorities

Contacts are spatially indexed by their COM and stored in chunks following a precomputed-like naming convention.

**Indexing (bounds, chunks) uses voxels at a specified resolution. Contact data (COM, faces, pointclouds) is stored in nanometers.**

## Contact Dataclass

```python
@attrs.frozen
class Contact:
    id: int
    seg_a: int
    seg_b: int
    com: Vec3D[float]  # center of mass in nm
    contact_faces: np.ndarray  # (N, 4) float32: x, y, z, affinity in nm
    local_pointclouds: dict[int, np.ndarray] | None  # segment_id -> (n_points, 3) in nm
    merge_decisions: dict[str, bool] | None  # authority -> yes/no
    partner_metadata: dict[int, Any] | None  # segment_id -> metadata

    def in_bounds(self, idx: VolumetricIndex) -> bool:
        """Check if COM falls within the given volumetric index."""
        ...

    def with_converted_coordinates(
        self, from_res: Vec3D, to_res: Vec3D
    ) -> Contact:
        """Return new Contact with coordinates converted between resolutions."""
        ...
```

## Info File Structure

The `info` JSON file at the dataset root:

```json
{
  "format_version": "1.0",
  "type": "contact",

  "resolution": [16, 16, 40],
  "voxel_offset": [0, 0, 0],
  "size": [6250, 6250, 1250],
  "chunk_size": [256, 256, 128],
  "max_contact_span": 512,

  "affinity_path": "gs://bucket/affinities",
  "segmentation_path": "gs://bucket/segmentation",
  "image_path": "gs://bucket/image",

  "local_point_clouds": [
    {"radius_nm": 200, "n_points": 1024},
    {"radius_nm": 2000, "n_points": 4096}
  ],

  "merge_decisions": ["ground_truth", "model_v1"],

  "filter_settings": {
    "min_seg_size_vx": 2000,
    "min_overlap_vx": 1000,
    "min_contact_vx": 5,
    "max_contact_vx": 2048
  }
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `format_version` | string | Format version for compatibility |
| `type` | string | Always `"contact"` |
| `resolution` | [x, y, z] | Voxel size in nanometers |
| `voxel_offset` | [x, y, z] | Dataset start in voxels |
| `size` | [x, y, z] | Dataset dimensions in voxels |
| `chunk_size` | [x, y, z] | Chunk dimensions in voxels |
| `max_contact_span` | int | Maximum contact span in voxels |
| `affinity_path` | string | Path to source affinity layer |
| `segmentation_path` | string | Path to source segmentation layer |
| `image_path` | string? | Optional path to image layer for visualization |
| `local_point_clouds` | array | Configurations for local point cloud sampling |
| `merge_decisions` | array | List of merge decision authority names |
| `filter_settings` | object | Filter parameters used during generation |

## Directory Structure

```
contact_dataset/
├── info
├── contacts/
│   ├── 0-256_0-256_0-128
│   ├── 256-512_0-256_0-128
│   └── ...
├── local_point_clouds/
│   ├── 200nm_1024pts/
│   │   ├── 0-256_0-256_0-128
│   │   └── ...
│   └── 2000nm_4096pts/
│       ├── 0-256_0-256_0-128
│       └── ...
└── merge_decisions/
    ├── ground_truth/
    │   ├── 0-256_0-256_0-128
    │   └── ...
    └── model_v1/
        └── ...
```

## Chunk Naming Convention

Follows precomputed format: `{x_start}-{x_end}_{y_start}-{y_end}_{z_start}-{z_end}`

Coordinates are in voxels at the specified resolution. For grid position `(gx, gy, gz)`:
```
x_start = voxel_offset[0] + gx * chunk_size[0]
x_end = x_start + chunk_size[0]
...
filename = f"{x_start}-{x_end}_{y_start}-{y_end}_{z_start}-{z_end}"
```

## Contact Assignment Rule

A contact is assigned to the chunk containing its **center of mass (COM)**. The `max_contact_span` constraint ensures contacts don't extend beyond what can be processed in a single operation.

## Binary Data Formats

All chunk files use a custom binary format with little-endian encoding.

### contacts/

Each chunk file contains all contacts whose COM falls within that chunk.

```
Header:
  - n_contacts: uint32 (number of contacts in chunk)

Per contact:
  - id: int64
  - seg_a: int64
  - seg_b: int64
  - com: float32[3] (x, y, z in nm)
  - n_faces: uint32
  - contact_faces: float32[n_faces, 4] (x, y, z, affinity per face)
```

### local_point_clouds/{radius}nm_{n_points}pts/

Each chunk contains point clouds for segments involved in contacts in that chunk.

```
Header:
  - n_entries: uint32

Per entry:
  - contact_id: int64
  - seg_a_points: float32[n_points, 3]
  - seg_b_points: float32[n_points, 3]
```

Points are sampled from segment meshes within a sphere of `radius_nm` around the contact COM.
The `n_points` is fixed per configuration (from info file).

### merge_decisions/{authority}/

Each chunk contains binary merge decisions for contacts in that chunk.

```
Header:
  - n_decisions: uint32

Per decision:
  - contact_id: int64
  - should_merge: uint8 (0 or 1)
```

## Reading Contacts

To read contacts in a bounding box:

1. Load `info` file
2. Calculate which chunks intersect the query bbox
3. For each chunk:
   - Load chunk file
   - Filter contacts whose COM is within query bbox
4. Optionally load corresponding local_point_clouds and merge_decisions

## Writing Contacts

Contacts are typically generated via a subchunkable operation:

1. Process each chunk with padding >= `max_contact_span / 2` (in voxels)
2. Find contacts, compute COM for each
3. Assign contacts to chunks based on COM
4. Write to appropriate chunk files

## Layer Architecture

Following the pattern of `VolumetricAnnotationLayer`:

### VolumetricContactLayer

```python
@attrs.frozen
class VolumetricContactLayer(Layer[VolumetricIndex, Sequence[Contact], Sequence[Contact]]):
    backend: ContactLayerBackend
    readonly: bool = False

    index_procs: tuple[IndexProcessor[VolumetricIndex], ...] = ()
    read_procs: tuple[ContactDataProcT, ...] = ()
    write_procs: tuple[ContactDataProcT, ...] = ()

    def __getitem__(self, idx: VolumetricIndex) -> Sequence[Contact]:
        ...

    def __setitem__(self, idx: VolumetricIndex, data: Sequence[Contact]):
        ...
```

### ContactLayerBackend

```python
@attrs.define
class ContactLayerBackend(Backend[VolumetricIndex, Sequence[Contact], Sequence[Contact]]):
    path: str
    resolution: Vec3D[int]  # voxel size in nm
    voxel_offset: Vec3D[int]  # dataset start in voxels
    size: Vec3D[int]  # dataset dimensions in voxels
    chunk_size: Vec3D[int]  # chunk dimensions in voxels
    max_contact_span: int  # in voxels
    # ... other info fields

    def read(self, idx: VolumetricIndex) -> Sequence[Contact]:
        ...

    def write(self, idx: VolumetricIndex, data: Sequence[Contact]):
        ...
```

## File Structure

```
zetta_utils/layer/volumetric/contact/
├── __init__.py
├── contact.py      # Contact dataclass
├── backend.py      # ContactLayerBackend
├── layer.py        # VolumetricContactLayer
└── build.py        # Builder functions
```

## Design Rationale

### Why COM-based assignment?
- Deterministic: each contact belongs to exactly one chunk
- Efficient queries: spatial indexing by a single point
- Avoids duplication across chunk boundaries

### Why max_contact_span constraint?
- Ensures contacts can be fully computed within a processing window
- Processing chunk must have padding >= max_contact_span / 2 (in voxels)
- Contacts exceeding this span are filtered out during generation

### Why separate folders for point clouds and decisions?
- Point clouds are expensive to compute/store, may not always be needed
- Multiple point cloud configurations (different radii) can coexist
- Merge decisions can come from multiple sources (ground truth, models)
- Each can be added/updated independently

### Why custom binary format?
- Compact storage for large datasets
- Direct memory mapping possible
- No external dependencies for reading/writing
- Variable-length contact_faces handled with per-contact n_faces field
