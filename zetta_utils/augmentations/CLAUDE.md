# Augmentations

Unified framework for data augmentation supporting both 2D and 3D tensor operations with probabilistic application.

## Core Components
Core Augmentation Operations (tensor.py): brightness_aug (scalar addition with optional masking), clamp_values_aug (value clamping with distribution-based thresholds), square_tile_pattern_aug (complex tile pattern generation with rotation support), apply_to_random_sections (applies functions to random Z-sections in 4D tensors), apply_to_random_boxes (applies functions to random 3D boxes with density/count control)

ImgAug Integration (imgaug.py): imgaug_readproc (wrapper for imgaug library integration), imgaug_augment (core function handling CXYZ/NXYC tensor conversion), auto-registration (automatically registers all imgaug augmenters 308+ functions), mode support (both 2D and 3D processing modes), multi-type support (images, heatmaps, segmentation maps, keypoints)

Misalignment Simulation (misalign.py): MisalignProcessor (simulates section misalignment for training robustness), two modes ("slip" for single section and "step" for cumulative displacement), integrated with volumetric indexing (works with VolumetricIndex for coordinate handling)

Probabilistic Framework (common.py): prob_aug (decorator for probabilistic augmentation application), universal wrapper (works with any augmentation function), flexible input (supports both positional and keyword arguments)

## Builder Registrations
Core Augmentations: brightness_aug → add_scalar_aug, clamp_values_aug → clamp_values_aug, square_tile_pattern_aug → square_tile_pattern_aug, apply_to_random_sections → apply_to_random_sections, apply_to_random_boxes → apply_to_random_boxes, imgaug_readproc → imgaug_readproc, MisalignProcessor → MisalignProcessor

ImgAug Integration: imgaug.augmenters.* → All imgaug augmenters auto-registered

## Usage Patterns
Training Specifications:
```cue
"@type": "imgaug_readproc"
"augmenters": [
    {
        "@type": "imgaug.augmenters.Sequential"
        "children": [{"@type": "imgaug.augmenters.Rot90", "k": [0, 1, 2, 3]}]
    }
]
```

Tensor Operations:
```python
# Direct usage in processing pipelines
square_tile_pattern_aug(data, tile_size=512, tile_stride=256, prob=1.0)

# Probabilistic application
@prob_aug(prob=0.5)
def my_augmentation(data):
    return processed_data
```

## Key Features
Probabilistic Application: All augmentations support prob parameter, distribution integration with zetta_utils.distributions, consistent probabilistic behavior across all augmentation types

Performance Optimization: Device-aware operations for GPU acceleration, efficient tensor handling for large volumes, optimized coordinate transformations

Type Safety: Comprehensive type annotations, runtime checking for tensor shapes and types, clear error messages for invalid inputs

## Development Guidelines
Creating Custom Augmentations: Use @prob_aug decorator for probabilistic application, register with builder system for spec usage, handle both 2D and 3D tensors where applicable, implement proper type checking and validation

Integration with ImgAug: Use imgaug_readproc for standard imgaug augmenters, handle tensor format conversions (CXYZ ↔ NXYC), consider mode compatibility (2D vs 3D), test augmentation pipelines thoroughly

Performance Considerations: Use efficient tensor operations, consider memory usage for large volumes, profile augmentation pipelines for bottlenecks, optimize for both CPU and GPU execution

Testing and Validation: Test with various tensor shapes and types, validate probabilistic behavior, check coordinate system consistency, verify augmentation correctness visually
