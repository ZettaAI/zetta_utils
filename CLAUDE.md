- never remove my breakpoints or uncomment code that i left commented out

# Rules
- Do what's asked, nothing more/less. NEVER create files unless absolutely necessary. ALWAYS prefer editing existing files.
- NEVER proactively create documentation files (*.md) or README files unless explicitly requested.
- NEVER add comments about what code used to be or what was moved/removed. Just make changes.
- NEVER import from scripts directory. Scripts import from main package, not vice versa.
- Follow instructions precisely. If asked to implement feature but not integrate, don't integrate. Do verbatim.
- NEVER use unittest mocks or any mocks OTHER THAN mocker fixture.

# Module Docs
**IMPORTANT**: Check `zetta_utils/{module_name}/CLAUDE.md` for module-specific architecture, patterns, guidelines.
**CRITICAL**: Update corresponding module CLAUDE.md after significant changes (new features, architecture changes, API changes, new builder registrations) to keep documentation current.

# Modules

Core (have detailed CLAUDE.md):
- task_management: PostgreSQL-backed distributed task management, multi-tenant, atomic assignment, dependency tracking
- mazepa: Distributed task execution framework, worker pools, task routing, execution checkpointing, error handling
- layer: Volumetric data backends abstraction, backend protocols, layer implementations, data access tools
- tensor_ops: Tensor operations/transformations, masking, normalization, label operations, multi-tensor support
- builder: Registry/factory system for dynamic object creation, type registration, building patterns, configuration-driven construction

Specialized:
- geometry: 3D geometry utilities for bounding boxes/vectors/spatial operations, bbox operations, striding, mask centering, vector math
- convnet: Convolutional neural network utilities/inference runners, model architectures, inference pipelines
- training: PyTorch training utilities/data loading, custom data loaders, samplers, training workflows
- augmentations: Data augmentation for training/inference, tensor augmentations, misalignment simulation, imgaug integration
- parsing: Configuration parsing/state management, CUE parsing, JSON handling, neuroglancer state management

Utility:
- common: Shared utilities/helper functions, context managers, path utilities, timers, signal handling
- db_annotations: Database-backed annotation management, collections, layers, annotation storage/retrieval
- api: External API interfaces, version management, API endpoints
- cli: Command-line interface tools, task management CLI, main entry points
- ng: Neuroglancer integration utilities, link builders, state management
- run: Runtime utilities/resource management, garbage collection, resource monitoring
- viz: Visualization utilities/widgets, rendering, interactive widgets

Data Processing:
- tensor_mapping: Tensor mapping/transformation utilities, coordinate mapping, tensor transformations
- message_queues: Message queue abstractions/serialization, queue interfaces, message serialization
- distributions: Distribution utilities for distributed computing, common distribution patterns
- mazepa_addons: Extensions/add-ons for mazepa framework, additional utilities for mazepa workflows
- mazepa_layer_processing: Layer processing operations for mazepa workflows, annotation postprocessing, operation protocols
- cloud_management: Cloud resource management utilities, cloud provider integrations

Guidelines: Module independence with minimal cross-dependencies, follow established patterns within each module, always check module-specific CLAUDE.md files for detailed guidance, comprehensive test coverage following module patterns
