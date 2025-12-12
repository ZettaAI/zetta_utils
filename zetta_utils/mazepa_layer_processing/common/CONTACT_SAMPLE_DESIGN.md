# ContactSampleOp Design

## Goal
Generate training samples for segment merge classifier from raw volumes.
Output: Arrow/Feather files (cross-language: Python + TypeScript).

## Inputs
- `candidate_layer`: Candidate segmentation (also used for meshes)
- `reference_layer`: Proofread reference segmentation
- `affinity_layer`: 3-channel affinity volume (X, Y, Z axes)

## Output Schema (Feather)
| Column | Type | Description |
|--------|------|-------------|
| `seg_a`, `seg_b` | int64 | Segment pair IDs |
| `should_merge` | int64 | 1=merge, 0=no merge |
| `n_contacts` | int64 | Actual contact count |
| `contacts` | list[list[float64]] | (max_contact_vx, 4) - [x, y, z, aff] in nm |
| `pointcloud_a`, `pointcloud_b` | list[list[float64]] | (n_points, 3) surface points in nm |
| `chunk_coord` | list[int64] | Chunk start coordinates (voxels) |
| `chunk_size` | list[int64] | Chunk dimensions (voxels) |
| `crop_pad` | list[int64] | Padding used (voxels) |
| `candidate_path` | string | Candidate segmentation path |
| `reference_path` | string | Reference segmentation path |
| `affinity_path` | string | Affinity volume path |

## Processing Steps

1. **Read volumes** (parallel) - candidate, proofread, affinity with padding

2. **Compute overlaps** - Between candidate segments and proofread connected components

3. **Filter bad segments** (BEFORE contact detection):
   - **Small**: total segment size < `min_seg_size_vx`
   - **Mergers**: overlap 2+ proofread CCs with >= `min_overlap_vx` each
   - **Unclaimed**: no proofread CC overlap >= `min_overlap_vx`

4. **Blackout** excluded segments (set to 0)

5. **Find contacts** - Detect voxel boundaries between remaining segments
   - Check X, Y, Z axes separately, use axis-specific affinity
   - Average affinities when voxel touches neighbor on multiple axes
   - Filter to kernel region (inside padding)

6. **Filter contact pairs**:
   - Low count (< `min_contact_vx`)
   - High count (> `max_contact_vx`)

7. **Download meshes** - Only for segments in valid pairs, clip to bbox

8. **Generate samples** per valid pair:
   - Compute affinity-weighted center of mass (COM)
   - Crop mesh points to sphere around COM (radius = min(crop_pad * resolution))
   - Sample `n_pointcloud_points` from each mesh (seed=42)
   - Label: 1 if both segments overlap same proofread CC, else 0
   - Pad contacts to fixed size

9. **Write feather** - Empty chunks produce files with 0 rows

## Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `output_path` | required | Output directory for feather files |
| `crop_pad` | (0,0,0) | Padding in voxels |
| `min_seg_size_vx` | 2000 | Min overlap voxels per segment |
| `min_overlap_vx` | 1000 | Min overlap for valid label |
| `min_contact_vx` | 5 | Min contacts per pair |
| `max_contact_vx` | 2048 | Max contacts (array size) |
| `n_pointcloud_points` | 2048 | Points per mesh |
