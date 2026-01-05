# DB Annotations

Database-backed annotation management system for storing and organizing neuroglancer annotations in Firestore.

## Architecture

- **Firestore Backend**: All data stored in Google Firestore with project-level configuration
- **Hierarchical Organization**: Collections → Layer Groups → Layers → Annotations
- **UUID-based IDs**: Deterministic UUIDs for annotations, random UUIDs for layers
- **Multi-user Support**: Created/modified tracking with user attribution
- **Neuroglancer Integration**: Native support for neuroglancer annotation types

## Core Models

### Annotations (annotation.py)
**AnnotationDBEntry**: Individual annotation storage with neuroglancer shape support
- **Supported Types**: Point, Line, Ellipsoid, AxisAlignedBoundingBox annotations
- **Metadata**: Collection/layer group association, comments, tags, selected segments
- **Timestamps**: Created/modified tracking with Unix timestamps
- **UUID Generation**: Deterministic UUID5 based on annotation content for deduplication

### Collections (collection.py)
**CollectionDBEntry**: Top-level organizational containers
- **Naming**: Human-readable names with lowercase indexing for search
- **User Tracking**: Created/modified by user attribution
- **Comments**: Optional descriptive text

### Layer Groups (layer_group.py)
**LayerGroupDBEntry**: Groups of related layers within collections
- **Hierarchy**: Belongs to collections, contains multiple layers
- **ID Format**: `{collection_id}:{name}` for unique identification
- **Layer Lists**: Array of layer IDs for grouping

### Layers (layer.py)
**LayerDBEntry**: Individual data layer definitions
- **Sources**: Path/URL to actual data layers
- **Validation**: Alphanumeric names with `-` and `_` only
- **UUID IDs**: Random UUID4 for unique identification

## Key Functions

### Annotation Management
```python
# Create annotations
add_annotation(annotation, collection_id, layer_group_id, comment, selected_segments, tags)
add_annotations(annotations_list, ...)  # Batch creation

# Query annotations
read_annotation(annotation_id)
read_annotations(annotation_ids=[], collection_ids=[], layer_group_ids=[], tags=[], union=True)

# Update/Delete
update_annotation(annotation_id, collection_id, layer_group_id, selected_segments, comment, tags)
delete_annotation(annotation_id)
```

### Builder Registrations
```python
# Convenience functions for common annotation types
@builder.register("add_point_annotation")
add_point_annotation(coord: Vec3D, collection_id, layer_group_id, ...)

@builder.register("add_bbox_annotation")
add_bbox_annotation(bbox: BBox3D, collection_id, layer_group_id, ...)
```

### Collection/Layer Group/Layer Operations
```python
# Collections
add_collection(name, user, comment)
read_collections(collection_ids=[])  # Query by IDs or get all
update_collection(collection_id, user, name, comment)

# Layer Groups
add_layer_group(name, collection_id, user, layers, comment)
read_layer_groups(layer_group_ids=[], collection_ids=[])
update_layer_group(layer_group_id, user, collection_id, name, layers, comment)

# Layers
add_layer(name, source, comment)
read_layers(layer_ids=[])
update_layer(layer_id, name, source, comment)
```

## Database Schema

### Indexed Columns (Fast Queries)
- **Annotations**: collection, layer_group, tags, created_at, modified_at
- **Collections**: name, name_lowercase, created_by, created_at, modified_by, modified_at
- **Layer Groups**: name, layers, collection, created_by, modified_by
- **Layers**: name, source

### Non-Indexed Columns (Storage Only)
- **Annotations**: comment, type, point, point_a, point_b, center, radii, selected_segments
- **Collections/Layer Groups/Layers**: comment

## Configuration & Constants
- **Project**: Uses `constants.DEFAULT_PROJECT` from zetta_utils
- **Database**: Environment variable `ANNOTATIONS_DB_NAME` (default: "annotations-fs")
- **Connection**: Automatic Firestore client initialization per database

## Usage Patterns

### Creating Hierarchical Structure
```python
# 1. Create collection
collection_id = add_collection("My Project", user="scientist1")

# 2. Create layer group
layer_group_id = add_layer_group(name="Experiment 1", collection_id=collection_id, user="scientist1")

# 3. Add annotations
add_point_annotation(
    coord=[100, 200, 50],
    collection_id=collection_id,
    layer_group_id=layer_group_id,
    comment="Interesting neuron"
)
```

### Querying with Filters
```python
# Get all annotations in specific collections
annotations = read_annotations(collection_ids=["project1", "project2"])

# Get annotations by tags (union=True means OR, union=False means AND)
tagged = read_annotations(tags=["validated", "complete"], union=False)

# Get layer groups for multiple collections
groups = read_layer_groups(collection_ids=["project1"])
```

## Development Guidelines

### Data Validation
- Layer names must match regex `[A-Za-z0-9_-]+`
- All user operations require user attribution
- Timestamps are automatically managed (Unix time)
- UUIDs ensure uniqueness and prevent duplicates

### Performance Considerations
- Use batch operations (`add_annotations`, `update_annotations`) for multiple items
- Index-aware queries for collection/layer_group/tag filtering
- Avoid large tag arrays (affects query performance)

### Error Handling
- `KeyError` for duplicate collection/layer group creation
- `ValueError` for invalid layer names
- Firestore exceptions bubble up for connection/permission issues

## Integration Points
- **Neuroglancer**: Direct support for viewer_state annotation types
- **Geometry Utils**: Vec3D and BBox3D for coordinate handling
- **Builder System**: Registered functions for config-driven usage
- **Layer System**: Integration with broader zetta_utils layer architecture
