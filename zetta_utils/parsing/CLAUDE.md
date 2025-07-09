# Parsing

Utilities for parsing configuration files and data formats including CUE, JSON, and Neuroglancer states.

## Core Components
CUE Parser (cue.py): Core Functions (load(), loads(), load_local()), External Dependency (requires CUE CLI tool cue export), Process (copies files to temp directory, validates with cue vet, exports to JSON), Features (supports local paths, remote files via fsspec, PosixPath objects)

JSON Parser (json.py): Enhanced JSON (custom encoder/decoder for Python tuples), Key Classes (ZettaSpecJSONEncoder, tuple_hook()), Features (preserves tuple types during serialization/deserialization), Functions (dumps(), loads(), dump(), load() with tuple support)

Neuroglancer State Parser (ngl_state.py): Purpose (parses neuroglancer annotation layers for ML training), Key Functions (read_remote_annotations(), write_remote_annotations()), Supported Types (BBox3D, Vec3D, PointAnnotation, AxisAlignedBoundingBoxAnnotation), Storage (cloud-based via CloudFiles, default: "gs://remote-annotations")

## Builder Integration
Configuration Loading: Primary Entry Point (builder.building.build(path=...) uses parsing.cue.load()), CLI Integration (zetta_utils run command parses CUE specs via CLI), Builder Registration (components register with @builder.register() decorators)

## Usage Patterns
Configuration Loading:
```python
# CUE file parsing
spec = cue.load("config.cue")
built_object = builder.build(spec)

# JSON with tuple support
data = json.loads(json_string)
```

State Management:
```python
# Load neuroglancer annotations
annotations = read_remote_annotations("layer_name")

# Save annotations
write_remote_annotations("layer_name", annotations)
```

Workflow Execution:
```cue
// CUE specification
{
    "@type": "some_operation"
    params: {
        bounds: (100, 200, 300)  // Tuple preserved
    }
}
```

## Key Dependencies
External Tools: CUE CLI (external CUE language tool for validation/export), fsspec (remote file system access), CloudFiles (cloud storage integration)

Internal Dependencies: neuroglancer (neuroglancer state objects), zetta_utils.geometry (BBox3D, Vec3D geometry types), zetta_utils.builder (builder system integration)

## Development Guidelines
CUE File Development: Use proper CUE syntax and validation, organize configurations in logical modules, use type definitions for complex structures, test CUE files with cue vet before use

JSON Handling: Use custom JSON functions for tuple support, be aware of type preservation requirements, handle nested complex structures properly, test serialization/deserialization roundtrips

Neuroglancer Integration: Use appropriate annotation types for data, handle cloud storage properly, consider annotation format compatibility, test with neuroglancer viewer

Performance Considerations: CUE parsing involves external process calls, cache parsed configurations when possible, consider file size for large configurations, use efficient JSON operations for large data
