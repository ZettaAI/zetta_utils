# Geometry

3D spatial operations and utilities for volumetric data processing.

## Core Components
Vec3D: Vec3D[T] (generic 3D vector with type-safe float/int variants), arithmetic operations (+, -, *, /, //, % with full operator support), comparison operators (proximity testing with isclose, allclose), NumPy integration (conversion utilities and array operations), precision control (VEC3D_PRECISION = 10 for coordinate rounding)

BBox3D: 3D axis-aligned bounding box (operations in nanometer units), construction (from slices, coordinates, points), spatial operations (intersection, union, containment, translation, padding, cropping), resolution-aware (coordinate transformations between different resolutions), grid operations (snapping and alignment operations), Neuroglancer integration (visualization support)

BBoxStrider: Efficient chunk iteration (over 3D volumes with stride patterns), three modes (shrink, expand, exact for boundary handling), superchunking (memory optimization for large chunk sets), multiprocessing support (for large chunk set processing), complex stride patterns (with offset support)

Mask Center: 2D binary mask analysis (centroid and interior point calculations), functions (centroid(), interior_point(), center_pole())

## Builder Registrations
Geometric Factories: BBox3D.from_slices, BBox3D.from_coords, BBox3D.from_points, BBoxStrider (class registration)

## Usage Patterns
Common Applications: Volumetric data chunking for distributed processing, spatial indexing for training datasets, coordinate system transformations between resolutions, bounding box operations for layer processing, mask analysis for structural analysis

Integration Points: Layer backends (CloudVolume, TensorStore, Annotation), Mazepa processing flows for chunked operations, training dataset indexers (volumetric, strided, NGL), Neuroglancer state management

## Development Guidelines
Vec3D Operations: Use type-safe variants (Vec3D[int], Vec3D[float]) for clarity, leverage arithmetic operator overloading for natural expressions, use isclose/allclose for floating-point comparisons, convert to/from NumPy arrays as needed

BBox3D Operations: Construct from appropriate factory methods based on input data, use resolution-aware transformations for coordinate system changes, handle boundary conditions carefully in spatial operations, leverage grid snapping for aligned operations

BBoxStrider Usage: Choose appropriate mode (shrink, expand, exact) based on requirements, use superchunking for memory-constrained environments, consider stride patterns for overlapping vs non-overlapping chunks, profile performance for large chunk sets

Performance Considerations: Vec3D operations are optimized for small-scale computations, BBox3D operations handle large coordinate values efficiently, BBoxStrider supports parallel processing for large datasets, use appropriate precision settings for coordinate operations
