# Layer

Unified abstraction for data access across different storage backends with type-safe indexing and processing pipelines.

## Core Abstractions
Layer Base (layer_base.py): Layer[BackendIndexT, BackendDataT, BackendDataWriteT] (generic layer with processing pipelines), index_procs/read_procs/write_procs (processing pipeline lists applied during operations), __getitem__/__setitem__ (delegate to read_with_procs/write_with_procs), probabilistic processing (JointIndexDataProcessor application based on probability)

Backend Abstraction (backend_base.py): Backend[IndexT, DataT, DataWriteT] (abstract base with read(idx), write(idx, data), with_changes(**kwargs)), VolumetricBackend (specialized for 3D data with caching, compression properties)

Processing Tools (tools_base.py): DataProcessor[T] (simple data transformation), IndexProcessor[T] (index transformation), JointIndexDataProcessor[DataT, IndexT] (coordinated index/data processing with probability), IndexChunker[IndexT] (index splitting for parallel processing)

## Key Implementations
Volumetric Layers (volumetric/): VolumetricLayer (primary layer for 3D data with resolution-aware indexing), VolumetricIndex (3D bounding box with resolution awareness, coordinate conversion), VolumetricBackend (abstract backend for volumetric data), Backends (CVBackend for CloudVolume, TSBackend for TensorStore, ConstantVolumetricBackend)

Database Layers (db_layer/): DBLayer (key-value database abstraction with flexible indexing), DBIndex (row-column index structure for database operations), Backends (FirestoreBackend, DatastoreBackend)

Layer Sets (layer_set/): LayerSet (manages multiple named layers as a single unit), VolumetricLayerSet (specialized for volumetric data collections)

Annotation Layers (volumetric/annotation/): VolumetricAnnotationLayer (handles geometric annotations: points, lines, bounding boxes), AnnotationBackend (backend for annotation data storage)

## Builder Registrations
Layer Builders: build_cv_layer (CloudVolume layer construction), build_ts_layer (TensorStore layer construction), build_db_layer (database layer construction), build_layer_set (multi-layer set construction), build_volumetric_layer_set (volumetric-specific layer set), build_annotation_layer (annotation layer construction), build_constant_volumetric_layer (constant value layer)

Processing Tools: VolumetricIndexTranslator (spatial translation), VolumetricIndexScaler (resolution scaling), VolumetricIndexOverrider (index parameter overriding), VolumetricIndexPadder (spatial padding), DataResolutionInterpolator (multi-resolution data interpolation), InvertProcessor (data inversion), ROIMaskProcessor (region-of-interest masking)

## Usage Patterns in Specs
Common Patterns: build_cv_layer for CloudVolume-based data access, build_ts_layer for TensorStore-based access, processor chains for data augmentation/resolution interpolation/spatial transformations, layer sets for managing related data streams

Data Pipeline Construction:
```cue
src: {
    "@type": "build_cv_layer"
    path: "gs://bucket/data"
    index_procs: [{"@type": "VolumetricIndexTranslator", offset: [100, 100, 0]}]
}
```

## Development Guidelines
Layer Construction: Use appropriate backend for storage type (CloudVolume, TensorStore, database), configure processing pipelines during layer creation, implement proper error handling for backend operations

Processing Pipelines: Chain processors in logical order (index → data → write), use probabilistic processors for augmentation during training, implement custom processors by inheriting from base processor protocols

Index Management: Use VolumetricIndex.from_coords() for coordinate-based indexing, handle resolution scaling and coordinate transformations properly, implement proper bounds checking for volumetric operations

Backend Implementation: Inherit from appropriate backend base class, implement required abstract methods (read, write, with_changes), add backend-specific properties and optimizations, handle caching and compression appropriately
