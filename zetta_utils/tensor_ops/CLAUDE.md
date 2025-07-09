# Tensor Operations

Comprehensive tensor manipulation utilities that work seamlessly with both PyTorch tensors and NumPy arrays.

## Core Operations
Mathematical Operations: Basic arithmetic (add, multiply, divide, int_divide, power, abs), comparison operations (compare with modes: eq, neq, gt, gte, lt, lte), tensor manipulation (rearrange, reduce, repeat - einops wrappers)

Shape Operations: squeeze/unsqueeze (dimension manipulation with intelligent handling), squeeze_to/unsqueeze_to (target-dimension shape adjustments), crop/crop_center (spatial cropping operations), pad_center_to (padding operations with multiple modes)

Interpolation: Advanced interpolate function supporting multiple modes (img, field, mask, segmentation), custom interpolation modes with tinybrain integration for segmentation downsampling, scale factor and size-based interpolation with validation

## Key Implementations
DictSupportingTensorOp Class: Core abstraction enabling operations on both single tensors and dictionaries of tensors, supports targeted operations with targets parameter, enables batch processing of multiple tensor channels

Type Conversion System: Seamless conversion between PyTorch tensors and NumPy arrays, dtype mapping with comprehensive handling of edge cases, device-aware operations with automatic memory management

Mask Operations: Connected component filtering (filter_cc, filter_cc3d), morphological operations via Kornia integration (erosion, dilation, opening, closing), advanced masking utilities with function composition

## Builder Registrations
Common Operations: rearrange, reduce, repeat, multiply, add, power, divide, int_divide, unsqueeze, squeeze, unsqueeze_to, squeeze_to, interpolate, compare, crop, crop_center, clone, tensor_op_chain, abs, pad_center_to

Conversion Operations: to_np, to_torch, to_float32, to_uint8

Generator Operations: get_affine_field, rand_perlin_2d, rand_perlin_2d_octaves

Label Operations: get_disp_pair, convert_seg_to_aff, seg_to_rgb

Mask Operations: filter_cc, filter_cc3d, kornia_opening, kornia_closing, kornia_erosion, kornia_dilation, mask_out_with_fn, combine_mask_fns

Multi-tensor Operations: compute_pixel_error, erode_combine

Normalization: apply_clahe

## Usage Patterns in Specs
Augmentation Pipelines:
```cue
processing: [
    {"@type": "rearrange", pattern: "c x y -> x y c"},
    {"@type": "interpolate", mode: "img", size: [512, 512]}
]
```

Mask Processing:
```cue
mask_processing: {"@type": "filter_cc", min_size: 100, connectivity: 6}
```

## Development Guidelines
Using DictSupportingTensorOp: Inherit from DictSupportingTensorOp for operations that should work on tensor dictionaries, implement apply_tensor() method for single tensor operations, use targets parameter to specify which dictionary keys to process

Type Conversion: Use to_torch/to_np for explicit conversions, operations handle mixed PyTorch/NumPy inputs automatically, be aware of device placement for PyTorch tensors

Performance Considerations: Use skip_on_empty_data decorator for operations that can skip empty tensors, consider memory usage for large 3D operations, batch operations when possible to reduce overhead

Custom Operations: Follow existing patterns for builder registration, implement proper error handling for edge cases, support both 2D and 3D operations where applicable
