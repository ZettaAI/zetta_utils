# Mazepa Layer Processing

Volumetric and layer-based processing operations built on top of mazepa. Bridges high-level CUE workflows with the mazepa execution engine — generates task graphs, manages memory via intermediaries, blends overlapping chunks, batches I/O, synchronizes reductions.

## Core Components
Operation Protocols (operation_protocols.py): VolumetricOpProtocol (3D volumetric ops with crop/padding: `get_input_resolution`, `with_added_crop_pad`, `__call__`, `make_task`), StackableVolumetricOpProtocol (extends VolumetricOpProtocol with `processing_fn` / `read` / `write` for batched I/O), ChunkableOpProtocol (generic op producing tasks), MultiresOpProtocol (resolution-aware input computation), ComputeFieldOpProtocol (field computation over source/target/destination layers)

Volumetric Apply Flow (common/volumetric_apply_flow.py): VolumetricApplyFlowSchema — master flow for chunked volumetric processing. Modes: simple direct write, with intermediaries (copy), checkerboarded + reduced (blended), deferred blending. Blend modes: linear, quadratic, bump, max.

Subchunkable Apply Flow (common/subchunkable_apply_flow.py): build_subchunkable_apply_flow (hierarchical multi-level flow with per-level blending and crop control; chains DelegatedSubchunkedOperation instances), build_postpad_subchunkable_apply_flow (wrapper that treats chunk sizes as post-padding bounds; simplifies config), DelegatedSubchunkedOperation (wraps a VolumetricApplyFlowSchema as a taskable op to enable hierarchical nesting)

Reduce Operations (common/reduce_operations.py): ReduceNaive (takes max in overlap regions — no blending), ReduceByWeightedSum (weighted-sum blending with linear/quadratic/bump weight functions)

Callable Operations (common/callable_operation.py, common/volumetric_callable_operation.py): CallableOperation (wraps `fn(layer_kwargs) → result` written to destination), VolumetricCallableOperation (wraps `fn(tensors) → tensor` with pad/crop, resolution scaling, per-fn semaphore support), build_chunked_callable_flow_schema, build_chunked_volumetric_callable_flow_schema

Stacked Volumetric Operation (common/stacked_volumetric_operation.py): StackedVolumetricOperation — batches multiple indices through a StackableVolumetricOpProtocol op; prefetches region, reads/processes/writes in bulk for I/O optimization.

Chunked Apply Flow (common/chunked_apply_flow.py): ChunkedApplyFlowSchema (simple chunking without blending; splits index, one task per chunk), build_chunked_apply_flow

Interpolate Flow (common/interpolate_flow.py): InterpolateOperation, build_interpolate_flow (multi-resolution interpolation)

Segment Contact (common/seg_contact_op.py): SegContactOp (mesh-based contact analysis with affinity scoring), AddPointcloudsOp (pointcloud aggregation)

Annotation Postprocessing (annotation_postprocessing.py): post_process_annotation_layer_flow

## Builder Registrations
Flow Builders: build_subchunkable_apply_flow, build_postpad_subchunkable_apply_flow, build_chunked_apply_flow, build_chunked_callable_flow_schema, build_chunked_volumetric_callable_flow_schema, build_interpolate_flow

Flow Schemas: VolumetricApplyFlowSchema, ChunkedApplyFlowSchema

Operations: CallableOperation, VolumetricCallableOperation, StackedVolumetricOperation, DelegatedSubchunkedOperation, InterpolateOperation, SegContactOp, AddPointcloudsOp

Reducers: ReduceNaive, ReduceByWeightedSum

Utilities: apply_mask_fn, write_fn, post_process_annotation_layer_flow

## Key Concepts
Subchunking: Break a large volumetric ROI into smaller processing chunks for memory fit and parallel execution.

Blending: When chunks overlap (due to padding), overlapping regions are reduced via weighted sum to avoid boundary artifacts. Modes: linear (ramp), quadratic (steeper at edges), bump (smooth), max (no blending), defer (skip reduction; leave intermediaries for external handling).

Cropping: Input padding supplies context; the pad is cropped from output to match expected size. Combined with blending for seamless tiling.

Checkerboarding: Multi-phase chunking where overlapping chunks are processed in separate phases, then reduced in a second pass. Avoids reading overlaps multiple times.

Intermediaries: Temporary layers (local `file://` or remote `gs://`) written by processing tasks and read by reduction tasks. Enables memory efficiency and distributed execution. Local intermediaries are auto-deleted after reduction.

Task Stacking: When an op implements StackableVolumetricOpProtocol, multiple indices can be batched into one task, reducing I/O overhead by reading/writing in bulk.

## Usage Patterns in Specs
Subchunkable Flow:
```cue
{
    "@type": "build_subchunkable_apply_flow"
    op: {"@type": "VolumetricCallableOperation", fn: {...}}
    dst: {"@type": "build_cv_layer", path: "gs://bucket/out"}
    processing_chunk_sizes: [[2048, 2048, 64], [512, 512, 32]]
    processing_crop_pads: [[0, 0, 0], [64, 64, 0]]
    processing_blend_pads: [[0, 0, 0], [32, 32, 0]]
    processing_blend_modes: ["max", "linear"]
    bbox: {...}
    dst_resolution: [4, 4, 40]
}
```

## Development Guidelines
Protocol Conformance: New volumetric ops should implement VolumetricOpProtocol (`get_input_resolution`, `with_added_crop_pad`, `__call__`, `make_task`). Use `@mazepa.taskable_operation_cls` for class-based ops; `@mazepa.flow_schema_cls` for flow schemas. Register with `@builder.register` for CUE accessibility.

Blending Constraints: `processing_blend_pad` must be ≤ `processing_chunk_size // 2`; all blended dims need `processing_chunk_size % 2 == 0` for halving. Checkerboarding (any non-`max` blend or crop) requires `intermediaries_dir` to be set; destination backend must support unaligned writes or enforce chunk alignment.

Reduction Chunk Size: Must be ≥ destination backend's chunk size (raises `ValueError` otherwise).

Resolution Handling: Respect `VolumetricIndex.resolution` during I/O. Operations that transform resolution (e.g., interpolation) must override `get_input_resolution(dst_resolution)` so flows can auto-scale reads.

Caching: Non-local backends can use `allow_cache=True` for LRU caching. `clear_cache_on_return` on VolumetricApplyFlowSchema clears all backend caches after flow completes.

Worker Routing & Semaphores: Use `op_worker_type` / `reduction_worker_type` to route tasks to named worker pools. Acquire `read` / `write` / `cuda` / `cpu` semaphores via layer backends; VolumetricCallableOperation supports `fn_semaphores` for fn-scope acquisition.

I/O Optimization: Implement StackableVolumetricOpProtocol (`processing_fn` / `read` / `write` separately) to enable `task_stack_size` batching.
