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

## Info File Structure

The `info` JSON file at the dataset root:

```json
{
  "format_version": "1.0",
  "type": "contact",

  "origin": [0, 0, 0],
  "resolution": [16, 16, 40],
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
| `origin` | [x, y, z] | World-space origin in voxels |
| `resolution` | [x, y, z] | Voxel size in nanometers |
| `chunk_size` | [x, y, z] | Chunk dimensions in voxels |
| `max_contact_span` | int | Maximum contact span in voxels (constraint) |
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

Coordinates are absolute voxel positions. For grid position `(gx, gy, gz)`:
```
x_start = origin[0] + gx * chunk_size[0]
x_end = x_start + chunk_size[0]
...
filename = f"{x_start}-{x_end}_{y_start}-{y_end}_{z_start}-{z_end}"
```

## Contact Assignment Rule

A contact is assigned to the chunk containing its **center of mass (COM)**. The `max_contact_span` constraint ensures contacts don't extend beyond what can be processed in a single operation.

## Data Formats

### contacts/

Each chunk file contains all contacts whose COM falls within that chunk.

Per contact:
- `id`: int64 - unique contact identifier
- `seg_a`: int64 - first segment ID
- `seg_b`: int64 - second segment ID
- `com`: float32[3] - center of mass in nanometers
- `contact_faces`: float32[N, 4] - (x, y, z, affinity) per face point

Serialization: Feather format with zstd compression.

### local_point_clouds/{radius}nm_{n_points}pts/

Each chunk contains point clouds for segments involved in contacts in that chunk.

Per contact:
- `contact_id`: int64 - references contact in contacts/
- `seg_a_points`: float32[n_points, 3] - sampled mesh points for seg_a
- `seg_b_points`: float32[n_points, 3] - sampled mesh points for seg_b

Points are sampled from segment meshes within a sphere of `radius_nm` around the contact COM.

### merge_decisions/{authority}/

Each chunk contains binary merge decisions for contacts in that chunk.

Per contact:
- `contact_id`: int64 - references contact in contacts/
- `should_merge`: bool - whether segments should merge

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

1. Process each chunk with padding >= `max_contact_span / 2`
2. Find contacts, compute COM for each
3. Assign contacts to chunks based on COM
4. Write to appropriate chunk files

## Design Rationale

### Why COM-based assignment?
- Deterministic: each contact belongs to exactly one chunk
- Efficient queries: spatial indexing by a single point
- Avoids duplication across chunk boundaries

### Why max_contact_span constraint?
- Ensures contacts can be fully computed within a processing window
- Processing chunk must have padding >= max_contact_span / 2
- Contacts exceeding this span are filtered out during generation

### Why separate folders for point clouds and decisions?
- Point clouds are expensive to compute/store, may not always be needed
- Multiple point cloud configurations (different radii) can coexist
- Merge decisions can come from multiple sources (ground truth, models)
- Each can be added/updated independently
