# Convnet

PyTorch-based neural network utilities for connectomics and computer vision workflows.

## Core Components
Model Management (utils.py): load_model() (supports loading models from JSON builder format, JIT, and ONNX formats), load_weights_file() (loads pre-trained weights with component filtering), save_model() (saves model state dictionaries), load_and_run_model() (convenience function for inference with automatic type conversion)

Architecture Components (architecture/): ConvBlock (flexible convolutional block with residual connections, normalization, and activation), UNet (complete U-Net implementation with skip connections sum/concat modes), Primitives (extensive collection of neural network building blocks)

Inference Runner (simple_inference_runner.py): SimpleInferenceRunner (cached inference runner with GPU memory management), supports 2D/3D operations, sigmoid activation, and zero-skipping optimizations

## Key Implementations
ConvBlock Architecture: Sequential convolutions with configurable activations and normalization, residual skip connections with pre/post-activation modes, flexible padding, stride, and kernel size configurations, supports both 2D and 3D convolutions

UNet Architecture: Traditional U-Net with encoder-decoder structure, configurable downsampling/upsampling operations, skip connections with sum or concatenation modes, built on ConvBlock foundation for consistency

Primitive Components: Tensor Operations (View, Flatten, Unflatten, Crop, CenterCrop), Pooling (MaxPool2DFlatten, AvgPool2DFlatten), Utilities (RescaleValues, Clamp, UpConv, SplitTuple), Multi-head (MultiHeaded, MultiHeadedOutput for multi-task learning)

## Builder Registrations
Core Components: ConvBlock (versioned >=0.0.2), UNet (versioned >=0.0.2), SimpleInferenceRunner, load_model, load_weights_file

PyTorch Integrations: All torch.nn.* classes auto-registered, All torch.optim.* classes auto-registered, PyTorch functional operations registered, custom builders for Sequential, Upsample, GroupNorm

Primitive Components: 15+ custom layer types registered, tensor manipulation utilities, pooling and cropping operations

## Usage Patterns
Model Loading:
```python
# Multi-format model loading with caching
model = load_model("model.json")  # Builder format
model = load_model("model.jit")   # TorchScript
model = load_model("model.onnx")  # ONNX format
```

Inference Workflows:
```python
# Batch processing with memory management
runner = SimpleInferenceRunner(
    model=model,
    device="cuda",
    skip_0_tiles=True
)
result = runner(input_tensor)
```

Architecture Definition:
```cue
// JSON-based model definition
model: {
    "@type": "UNet"
    layers_enc: [32, 64, 128]
    layers_dec: [128, 64, 32]
    skip_mode: "concat"
}
```

## Development Guidelines
Model Development: Use ConvBlock for consistent convolutional operations, leverage UNet for encoder-decoder architectures, register custom architectures with builder system, use versioned registration for backward compatibility

Inference Optimization: Use SimpleInferenceRunner for batch processing, enable zero-skipping for sparse data, configure appropriate device placement, monitor GPU memory usage

Architecture Composition: Build complex models from primitive components, use builder system for configuration-driven development, implement proper error handling for model loading, cache frequently loaded models

Performance Considerations: Cached model loading (LRU cache with maxsize=2), memory management with periodic cleanup, type-safe tensor operations with automatic conversion, GPU memory optimization for large models
